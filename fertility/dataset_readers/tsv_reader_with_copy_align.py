import csv
from collections import Counter
from typing import Dict, Optional, List
import logging
import copy

import numpy as np
from allennlp.common import Lazy
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from tqdm import tqdm

from fertility.dataset_readers.preprocessor import Preprocessor, is_number
from fertility.decoding.decoding_grammar import DecodingGrammar
from fertility.constants import COGS_VAR, COPY_SYMBOL

logger = logging.getLogger(__name__)

def get_indexing(english):
    """
    Returns two data structures that help mapping strings to unique ids and back.
    """
    evocab = set(word for sent in english for word in sent)
    i2e = sorted(evocab) #consistent ordering
    e2i = { w : i for i,w in enumerate(i2e)}
    return i2e,e2i

try:
    # This is much, much faster!
    import numba

    @numba.njit()
    def add_soft_counts(counts, lexp, e_ids, f_ids):
        Zs = np.zeros(len(f_ids))
        matrix = np.zeros((len(e_ids), len(f_ids)))

        for i, e_id in enumerate(e_ids):
            for j, f_id in enumerate(f_ids):
                matrix[i, j] = lexp[f_id, e_id]
                Zs[j] += lexp[f_id, e_id]

        matrix /= Zs  # normalize columns by their individual sums

        # add pseudo-counts to overall pseudo-counts
        for i, e_id in enumerate(e_ids):
            for j, f_id in enumerate(f_ids):
                counts[f_id, e_id] += matrix[i, j]

except ModuleNotFoundError:

    def add_soft_counts(counts, lexp, e_ids, f_ids):
        # matrix is an I x J matrix, and contains the subset of lexical probabilities for all word-to-word translations
        # in other words: matrix[i,j] is the probability that the English word at index j translates to the French word at index i
        matrix = np.array([[lexp[f, e] for f in f_ids] for e in e_ids])

        Zs = np.sum(matrix, axis=0) #sums of columns
        matrix /= Zs #normalize columns by their individual sums

        for i, e_id in enumerate(e_ids): #add pseudo-counts to overall pseudo-counts
            for j, f_id in enumerate(f_ids):
                counts[f_id, e_id] += matrix[i, j]


class IBM1:
    """
    IBM1 model with a extras that can optionally be turned on.
    Follows the terminology in Adam Lopez' note, i.e. the lexical translation probabilities are P(f|e)
    """

    def __init__(self, english, french, init_same=0):
        """
        Takes as argument two lists of lists one for each language.

        Translation direction: English -> French

        Options:
            null_alignment: do we allow that a french word is aligned to no English word?
            init_same: give words that have same string on English and French side the given pseudo-count on initialization. 0 means uniform init.
            entropy_trick: do you want to activate a theoretically not justified trick that encourages that words tend to have few translations?
        """
        self.i2e, self.e2i = get_indexing(english)  # maps id -> word and back
        self.i2f, self.f2i = get_indexing(french)  # we don't need a French null token (we model P(f|e))
        self.I = len(self.i2f)  # number of French words
        self.J = len(self.i2e)  # number of English words
        # lexp: I x J matrix
        # lexp_ij = P(f_i|e_j)
        # because of probability all columns must sum to 1.
        self.lexp = np.ones((self.I, self.J),
                            dtype="float32")  # initialize counts uniformly, 32 bit floats should be precise enough
        if init_same > 0:
            for ew, j in self.e2i.items():
                for fw, i in self.f2i.items():
                    if fw == ew:
                        self.lexp[i, j] += init_same  # give higher counts to words with same string
        column_sums = np.sum(self.lexp, axis=0)
        self.lexp /= column_sums  # normalize counts to form probability distribution.

        self.english = english
        self.french = french
        self.init_same = init_same

        self.en_ids = [np.array([self.e2i[w] for w in sent]) for sent in self.english]
        self.f_ids = [np.array([self.f2i[w] for w in sent]) for sent in self.french]

    def one_EM_step(self):
        """
        Performs an E-step followed by an EM step. Reestimates self.lexp.
        """
        counts = np.zeros((self.I, self.J),
                          dtype="float32")  # will contain all expected counts. After renormalization it will be the new self.lexp
        # E step
        for e_ids, f_ids in zip(self.en_ids, self.f_ids):
            add_soft_counts(counts, self.lexp, e_ids, f_ids)

        # M step
        margin_counts = np.sum(counts, axis=0)  # vector of length J, contains column sums
        counts /= margin_counts
        self.lexp = counts

    def EM(self, n):
        """
        Performs EM for n rounds.
        """
        for i in tqdm(range(n)):
            # print("EM round {}".format(i + 1), file=sys.stderr)
            self.one_EM_step()

    def posterior(self, english, french):
        e_ids = [self.e2i[w] for w in english]
        f_ids = [self.f2i[w] for w in french]
        # matrix is an I x J matrix, and contains the subset of lexical probabilities for all word-to-word translations
        # in other words: matrix[i,j] is the probability that the English word at index j translates to the French word at index i
        matrix = np.array([[self.lexp[f, e] for f in f_ids] for e in e_ids])

        Zs = np.sum(matrix, axis=0)  # sums of columns
        matrix /= Zs  # normalize columns by their individual sums

        return matrix

    def decode_posterior(self, english, french, min_prob):
        # Returns a list tuples of the format (source idx, target idx)
        m = self.posterior(english, french).transpose()
        r = []
        for i, f in enumerate(m):
            if max(f) >= min_prob:
                r.append((np.argmax(f), i))
        return r

    def decode(self, english, french, filter_out_null=True):
        """
        Finds the highest scoring alignment (for each french word); this is the Viterbi alignment.
        Returns a list of tuples where the first component is an index on the french side and the second belongs to the English side.
        """
        e_ids = [self.e2i[w] for w in english]
        f_ids = [self.f2i[w] for w in french]
        # for each french word pick the english word such that P(f|e) is maximal
        a = [np.argmax([self.lexp[f, e] for i, e in enumerate(e_ids)]) for j, f in enumerate(f_ids)]
        # now return a list of tuples (french,english)
        return list(enumerate(a))

    def print_decode(self, english, french):
        """
        Find most Viterbi alignment between english and french sentence and print the output.
        """
        a = self.decode(english, french)
        o = []
        for f, e in a:
            o.append("{}-{}".format(f, e))
        print(" ".join(o))





@DatasetReader.register("tsv_reader_with_copy_align")
class Seq2SeqDatasetReader(DatasetReader):
    """
    Read a tsv file containing paired sequences, and create a dataset suitable for a
    `ComposedSeq2Seq` model, or any model with a matching API.

    Expected format for each input line: <source_sequence_string>\t<target_sequence_string>

    The output of `read` is a list of `Instance` s with the fields:
        source_tokens : `TextField` and
        target_tokens : `TextField`

    `START_SYMBOL` and `END_SYMBOL` tokens are added to the source and target sequences.

    # Parameters

    source_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to `SpacyTokenizer()`.
    target_tokenizer : `Tokenizer`, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to `source_tokenizer`.
    source_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define input (source side) token representations. Defaults to
        `{"tokens": SingleIdTokenIndexer()}`.
    target_token_indexers : `Dict[str, TokenIndexer]`, optional
        Indexers used to define output (target side) token representations. Defaults to
        `source_token_indexers`.
    source_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `start_symbol` to the beginning of the source sequence.
    source_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `end_symbol` to the end of the source sequence.
    target_add_start_token : `bool`, (optional, default=`True`)
        Whether or not to add `start_symbol` to the beginning of the target sequence.
    target_add_end_token : `bool`, (optional, default=`True`)
        Whether or not to add `end_symbol` to the end of the target sequence.
    start_symbol : `str`, (optional, default=`START_SYMBOL`)
        The special token to add to the end of the source sequence or the target sequence if
        `source_add_start_token` or `target_add_start_token` respectively.
    end_symbol : `str`, (optional, default=`END_SYMBOL`)
        The special token to add to the end of the source sequence or the target sequence if
        `source_add_end_token` or `target_add_end_token` respectively.
    delimiter : `str`, (optional, default=`"\t"`)
        Set delimiter for tsv/csv file.
    quoting : `int`, (optional, default=`csv.QUOTE_MINIMAL`)
        Quoting to use for csv reader.
    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        real_target_token_indexers: Dict[str, TokenIndexer] = None,
        non_copyable: Optional[List[str]] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = False,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        em_epochs: int = 80,
        delimiter: str = "\t",
        copy_despite_case_mismatch : bool = False,
        quoting: int = csv.QUOTE_MINIMAL,
        copy: bool = True,
        grammar: Optional[Lazy[DecodingGrammar]] = None,  # This may be used to restrict copying to symbols that may appear on the target side.
        preprocessor: Optional[Preprocessor] = None,
        add_type_count: bool = False,
        enable_cogs_var: bool = False,
        pass_real_targets: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.em_epochs = em_epochs
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._real_target_token_indexers = real_target_token_indexers or self._source_token_indexers

        self.preprocessor = preprocessor or Preprocessor.by_name("identity")()

        self.add_type_count = add_type_count

        self.pass_real_targets = pass_real_targets

        self.copy_despite_case_mismatch = copy_despite_case_mismatch
        self._source_add_start_token = source_add_start_token
        self._end_token: Optional[Token] = None
        self.copy = copy
        # if (
        #     source_add_start_token
        # ):
        #     if source_add_start_token:
        #         self._check_start_end_tokens(start_symbol, self._source_tokenizer)
        #
        # if source_add_end_token:
        #     self._check_start_end_tokens(end_symbol, self._source_tokenizer)
        self._source_add_end_token = source_add_end_token

        self._start_token = start_symbol
        self._end_token = end_symbol

        self.enable_cogs_var = enable_cogs_var

        self._delimiter = delimiter
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        # This represents the set of input tokens that MUST NOT be copied
        self.non_copyable = set(non_copyable) if non_copyable is not None else set()

        # If this is None, all other tokens may be copied
        self.copyable_inputs = None
        if grammar:
            # If we have a grammar, we can exclude some tokens for copying (at test time) because they're not covered.
            grammar = grammar.construct(tok2id=dict()).grammar
            self.copyable_inputs = set()
            for prod in grammar.productions():
                for x in prod.rhs():
                    if isinstance(x, str):
                        self.copyable_inputs.add(x)
            self.copyable_inputs = self.copyable_inputs - self.non_copyable

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        sources_tokenized = []
        targets_tokenized = []
        categories = []
        targets_raw= []
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in self.shard_iterable(
                enumerate(csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting))
            ):
                if len(row) != 2 and len(row) != 3:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                if len(row) == 3:
                    source_sequence, target_sequence, category = self.preprocessor.preprocess(row)
                else:
                    source_sequence, target_sequence = self.preprocessor.preprocess(row)
                    category = None

                if len(source_sequence) == 0 or len(target_sequence) == 0:
                    continue


                source_toks = [x.text for x in self._source_tokenizer.tokenize(source_sequence)]

                if self._source_add_start_token:
                    source_toks.insert(0, self._start_token)

                if self._source_add_end_token:
                    source_toks.append(self._end_token)

                sources_tokenized.append(source_toks)
                targets_tokenized.append([x.text for x in self._target_tokenizer.tokenize(target_sequence)])
                targets_raw.append(target_sequence)
                categories.append(category)

        aligner = IBM1(sources_tokenized, targets_tokenized)

        aligner.EM(self.em_epochs)

        index = 0
        for s, t, r, cat in zip(sources_tokenized, targets_tokenized, targets_raw, categories):
            # print(s,t, [(s[y], t[x]) for y,x in aligner.decode_posterior(s,t, 0.7)])
            yield self.text_to_instance(s, t, r, index, aligner.posterior(s, t), cat)
            index += 1
                # yield self.text_to_instance(source_sequence, target_sequence)

    def text_to_instance(
        self, source_toks: List[str], target_toks: List[str],
            target_string: str, index: int,
            alignment_matrix: np.array,
            category: Optional[str] = None
    ) -> Instance:  # type: ignore
        source_field = TextField([Token(t) for t in source_toks])
        d = {"source_tokens": source_field}
        metadata = {"source_tokens":source_toks, "#": index}
        if category:
            metadata["category"] = category

        assert alignment_matrix.shape == (len(source_toks), len(target_toks))
        if self.enable_cogs_var:
            # aligner might have aligned variables to something, which is most likely inconsistent with
            # our assumption that we know where the variables align, so zero those probabilities out
            for i, tt in enumerate(target_toks):
                if is_number(tt):
                    alignment_matrix[:, i] = 0.0
                    alignment_matrix[int(tt) + int(self._source_add_start_token), i] = 1.0

        d["alignment"] = TensorField(alignment_matrix)

        if self.copyable_inputs is not None and self.copy:
            copyable_inputs = np.zeros((len(metadata["source_tokens"])))
            for i, tok in enumerate(metadata["source_tokens"]):
                if tok in self.copyable_inputs:
                    copyable_inputs[i] = 1.0
            d["copyable_inputs"] = TensorField(copyable_inputs)

        if target_toks is not None:
            metadata["target_tokens"] = target_toks
            metadata["target_string"] = target_string

            if self.add_type_count:
                counter = Counter(target_toks)
                target_types = []
                target_counts = []
                max_freq = max(counter.values())
                for typ, freq in counter.items():
                    target_types.append(typ)
                    a = np.zeros(max_freq+1, dtype=np.bool)
                    for i in range(freq+1):
                        a[i] = True
                    target_counts.append(a)

                d["target_types"] = TextField([Token(x) for x in target_types])
                d["target_counts"] = TensorField(np.array(target_counts))

            if not self.copy:
                my_target_toks = target_toks
            else:
                if self.pass_real_targets:
                    if self.enable_cogs_var:
                        # giving the var. posterior the variables as numbers makes it probably harder for the posterior
                        # so give it the corresponding input token.
                        # d["real_target_tokens"] = TextField([Token(x) for x in target_toks])
                        d["real_target_tokens"] = TextField([Token(x) if not is_number(x) else Token(source_toks[int(x) + int(self._source_add_start_token)]+"-VAR") for x in target_toks])
                    else:
                        d["real_target_tokens"] = TextField([Token(x) for x in target_toks])

                my_target_toks = []
                copy_mask = np.zeros((len(metadata["source_tokens"]), len(metadata["target_tokens"])))
                for i, target_tok in enumerate(metadata["target_tokens"]):
                    found = None
                    if target_tok not in self.non_copyable:
                        for k, source_tok in enumerate(metadata["source_tokens"]):
                            if source_tok == target_tok or (self.copy_despite_case_mismatch and source_tok.lower() == target_tok.lower()):
                                copy_mask[k, i] = 1.0
                                found = COPY_SYMBOL

                        if self.enable_cogs_var and is_number(target_tok):
                            found = COGS_VAR
                            copy_mask[int(target_tok) + int(self._source_add_start_token), i] = 1.0
                            #COGS variable can only be introduced by the token with the correct index.

                    if found is not None:
                        my_target_toks.append(found)
                    else:
                        my_target_toks.append(target_tok)

                    d["source_to_copy_mask"] = TensorField(copy_mask)

            target_field = TextField([Token(x) for x in my_target_toks])
            d["target_tokens"] = target_field


        d["metadata"] = MetadataField(metadata)
        return Instance(d)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore
        if "target_types" in instance.fields:
            instance.fields["target_types"]._token_indexers = self._real_target_token_indexers

        if "real_target_tokens" in instance.fields:
            instance.fields["real_target_tokens"]._token_indexers = self._real_target_token_indexers

    # def _check_start_end_tokens(
    #     self, start_symbol: str, tokenizer: Tokenizer
    # ) -> None:
    #     """Check that `tokenizer` correctly appends `start_symbol` and `end_symbol` to the
    #     sequence without splitting them. Raises a `ValueError` if this is not the case.
    #     """
    #
    #     tokens = tokenizer.tokenize(start_symbol)
    #     err_msg = (
    #         f"Bad start or end symbol ('{start_symbol}') "
    #         f"for tokenizer {self._source_tokenizer}"
    #     )
    #     try:
    #         start_token = tokens[0]
    #     except IndexError:
    #         raise ValueError(err_msg)
    #     if start_token.text != start_symbol:
    #         raise ValueError(err_msg)
