import csv
import itertools
from collections import Counter, defaultdict
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

from fertility.dataset_readers.lexicon_learning import Lexicon
from fertility.dataset_readers.preprocessor import Preprocessor, is_number
from fertility.dataset_readers.tsv_reader_with_copy_align import IBM1
from fertility.decoding.decoding_grammar import DecodingGrammar
from fertility.constants import COGS_VAR, COPY_SYMBOL

logger = logging.getLogger(__name__)


def get_match(lexicon, s, target_items):
    for i, t in enumerate(target_items):
        if lexicon.is_match(s, t):
            return i
    return None


@DatasetReader.register("tsv_reader_with_copy_align_counter")
class Seq2SeqDatasetReaderCounter(DatasetReader):
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
        # target_token_indexers: Dict[str, Lazy[TokenIndexer]] = None,
        non_copyable: Optional[List[str]] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = False,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        em_epochs: int = 80,
        delimiter: str = "\t",
        quoting: int = csv.QUOTE_MINIMAL,
        copy: bool = True,
        lexicon: Optional[Lexicon] = None,
        preprocessor: Optional[Preprocessor] = None,
        enable_cogs_var : bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self.em_epochs = em_epochs
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}

        self._target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="target_tokens")}

        self.preprocessor = preprocessor or Preprocessor.by_name("identity")()

        self.lexicon = lexicon

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


        self._delimiter = delimiter
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        # This represents the set of input tokens that MUST NOT be copied
        self.non_copyable = set(non_copyable) if non_copyable is not None else set()

        # If this is None, all other tokens may be copied
        self.copyable_inputs = None

        self.enable_cogs_var = enable_cogs_var

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


            if not self.copy:
                my_target_toks = target_toks
            else:
                my_target_toks = []
                target_set = set(metadata["target_tokens"])
                target_counts = Counter(metadata["target_tokens"])
                source_set = set(metadata["source_tokens"])

                items_copy = set()
                items_cogs_var = set()
                item_frequency = dict()
                for t in metadata["target_tokens"]:
                    # if t not in self.non_copyable and t in source_set:
                    if any(self.lexicon.is_match(s, t) for s in source_set):
                        items_copy.add(t)
                        my_target_toks.append(COPY_SYMBOL)
                    elif self.enable_cogs_var and is_number(t):
                        items_cogs_var.add(t)
                        my_target_toks.append(COGS_VAR)
                    else:
                        my_target_toks.append(t)

                for item in itertools.chain(items_copy, items_cogs_var):
                    item_frequency[item] = target_counts[item]
                    # del target_counts[item]

                items_copy = sorted(items_copy)
                items_cogs_var = sorted(items_cogs_var)

                rule_mask_copy = np.zeros((len(source_toks), len(items_copy)), dtype=bool)
                rule_mask_cogs_var = np.zeros((len(source_toks), len(items_cogs_var)), dtype=bool)
                for i, input_tok in enumerate(metadata["source_tokens"]):
                    # if input_tok in target_set and input_tok not in self.non_copyable:
                    match = get_match(self.lexicon, input_tok, items_copy)
                    if match is not None:
                        rule_mask_copy[i, match] = True

                if self.enable_cogs_var:
                    for idx, t in enumerate(items_cogs_var):
                        rule_mask_cogs_var[int(t) + int(self._source_add_start_token), idx] = True

                d["rule_mask_copy"] = TensorField(rule_mask_copy)
                d["rule_freq_copy"] = TensorField(
                    np.array([item_frequency[item] for item in items_copy], dtype=np.long))

                if self.enable_cogs_var:
                    d["rule_mask_cogs_var"] = TensorField(rule_mask_cogs_var)
                    d["rule_freq_cogs_var"] = TensorField(np.array([item_frequency[item] for item in items_cogs_var], dtype=np.long))

            target_field = TextField([Token(x) for x in my_target_toks])
            d["target_tokens"] = target_field

        d["metadata"] = MetadataField(metadata)
        return Instance(d)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore


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
