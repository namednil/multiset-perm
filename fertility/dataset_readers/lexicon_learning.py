from collections import defaultdict

import sys
from collections import Counter
import pdb
import pickle, json
from typing import List

from allennlp.common import Registrable
from allennlp.data import Tokenizer

from fertility.dataset_readers.preprocessor import Preprocessor

EPS=3 #3

def ekins_lexicon_learner(input_strs: List[List[str]], output_strs: List[List[str]]):
    """
    Simple Lexicon Learning rule from Akyürek and Andreas 2021, "Lexicon Learning for Few-Shot Neural Sequence Modeling".
    :param input_strs:
    :param output_strs:
    :return:
    """
    word_alignment = {}
    inputs  = []
    outputs = []

    # BEGIN EKIN'S CODE
    # SEE https://github.com/ekinakyurek/lexical/blob/e7a44e19d23a1d99726cd76c5cd88f56ca586653/utils/summarize_aligned_data3.py
    for input, output in zip(input_strs, output_strs):
        input, output = set(input), set(output) # CHANGE: removed tokenization
        inputs.append(input)
        outputs.append(output)
        for inp in input:
            if inp not in word_alignment:
                word_alignment[inp] = {}
            inpmap = word_alignment[inp]
            for out in output:
                if out not in inpmap:
                    inpmap[out] = 1
                else:
                    inpmap[out] = inpmap[out] + 1

    for i in range(len(inputs)):
        input = inputs[i]
        output = outputs[i]
        for k in input:
            for v in list(word_alignment[k].keys()):
                if v not in output:
                    del word_alignment[k][v]
            # else:
            #     for v in list(word_alignment[k].keys()):
            #         if v in outputs[i]:
            #             del word_alignment[k][v]

    incoming = {}
    for (k,mapped) in list(word_alignment.items()):
        for (v,_) in mapped.items():
            if v in incoming:
                incoming[v].add(k)
            else:
                incoming[v] = {k,}

        # if len(word_alignment[k]) == 0:
        #     del word_alignment[k]
    # print(incoming["9"])
    for (v, inset) in incoming.items():
        if len(inset) > EPS:
            # print(f"common word: v: {v}, inset: {inset}")
            # print("deleting ", v)
            for (k,mapped) in list(word_alignment.items()):
                if v in mapped:
                    # print(f"since EPS deleting {k}->{v}") # CHANGE: commented this out
                    del word_alignment[k][v]

    for (v,inset) in incoming.items():
        if len(inset) > 1:
            candidates = set([e for e in inset])
            # for k, line in enumerate(data):
            for input, output in zip(input_strs, output_strs): # CHANGE: changed this line, as reading from file and splitting not necessary
                if len(candidates) == 0:
                    break
                # input, output = line.split(SPLIT_TOK) # CHANGE
                input, output = set(input), set(output) # CHANGE: removed tokenization
                if v in output:
                    for e in set(candidates):
                        if e not in input:
                            candidates.remove(e)
            if len(candidates) == 1:
                wrongs = inset-candidates
                for t in wrongs:
                    if v in word_alignment[t]:
                        # print(f"in candidates deleting {t}->{v}") # CHANGE: commented this out
                        del word_alignment[t][v]

    for (k,mapped) in list(word_alignment.items()):
        if len(word_alignment[k]) == 0:
            del word_alignment[k]
        else:
            if k in mapped:
                mapped[k] += 1

    # END EKIN'S CODE
    # word_alignment is a dictionary mapping input tokens to a dictionary of output tokens mapping them onto counts
    # We extract a lexicon entry pair (a,b) if a is uniquely mapped to an output token

    lexicon = {k: list(word_alignment[k])[0] for k in word_alignment if len(word_alignment[k]) == 1}
    return lexicon



class Lexicon(Registrable):

    def __contains__(self, item):
        raise NotImplementedError()
    def __getitem__(self, item):
        raise NotImplementedError()

    def is_match(self, source: str, target: str) -> bool:
        raise NotImplementedError()


@Lexicon.register("copy_lexicon")
class CopyLexicon(Lexicon):
    def __init__(self,  non_copyable: List[str] = None):
        self.non_copyable = non_copyable or []

    def __contains__(self, item):
        return item not in self.non_copyable

    def __getitem__(self, item):
        if item in self.non_copyable:
            raise KeyError(f"{item} cannot be copied.")
        return item

    def is_match(self, source: str, target: str) -> bool:
        return source == target and target in self


@Lexicon.register("simple_lexicon")
class SimpleLexicon(Lexicon):

    def __init__(self, filename, source_tokenizer: Tokenizer, target_tokenizer: Tokenizer,
                 copy: bool = True,
                 preprocessor: Preprocessor = None,
                 copy_despite_case_mismatch: bool = False,
                 non_copyable: List[str] = None):
        self.non_copyable = non_copyable or []
        self.copy = copy
        self.copy_despite_case_mismatch = copy_despite_case_mismatch
        self.preprocessor = preprocessor or Preprocessor.by_name("identity")()
        with open(filename) as f:
            source = []
            targets = []
            for line in f:
                content = line.strip().split("\t")
                content = self.preprocessor.preprocess(content)
                nl, lf = content[0], content[1]
                source.append([x.text for x in source_tokenizer.tokenize(nl)])
                targets.append([x.text for x in target_tokenizer.tokenize(lf)])
            self.lexicon = ekins_lexicon_learner(source, targets)


    def __contains__(self, item):
        if self.copy:
            if item not in self.non_copyable:
                return True
        return item in self.lexicon

    def is_match(self, source: str, target: str) -> bool:
        if source in self.lexicon:
            if self.copy_despite_case_mismatch and self.lexicon[source].lower() == target.lower():
                return True
            return self.lexicon[source] == target

        if self.copy:
            if self.copy_despite_case_mismatch:
                return source.lower() == target.lower() and target not in self.non_copyable
            else:
                return source == target and target not in self.non_copyable
        return False

    def __getitem__(self, item):
        if item in self.lexicon:
            return self.lexicon[item]
        if self.copy:
            if self.non_copyable is not None and item in self.non_copyable:
                raise KeyError()
            return item
        else:
            raise KeyError()




if __name__ == "__main__":
    with open("../../data/atis/atis_funql_train_brackets.tsv") as f:
        source = []
        targets = []
        for line in f:
            nl, lf = line.strip().split("\t")
            # nl, lf, _ = line.strip().split("\t")
            source.append(nl.split(" "))
            targets.append(lf.split(" "))

        lexicon = ekins_lexicon_learner(source, targets)

        print(lexicon)
        # print({k:v for k,v in lexicon.items() if k != v and k[:-1] != v})