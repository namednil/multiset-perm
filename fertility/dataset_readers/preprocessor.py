from typing import List

from allennlp.common import Registrable


class Preprocessor(Registrable):

    def preprocess(self, row: List[str]) -> List[str]:
        raise NotImplementedError()

    def postprocess(self, logical_form: List[str]) -> List[str]:
        raise NotImplementedError()


@Preprocessor.register("identity")
class IdentityPreprocessor(Preprocessor):

    def preprocess(self, row: List[str]) -> List[str]:
        return row

@Preprocessor.register("cogs")
class COGSPreprocessor(Preprocessor):
    useless_tokens = {".", ",", "_", "(", ")", "x"} # these tokens don't add anything semantically and could be reconstructed based on simple rules.

    def preprocess(self, row: List[str]) -> List[str]:

        nl, lf, category = row

        nl = nl.strip(".").strip() # important: remove ., so the model is not forced to copy . in the outputs!

        lf = lf.split(" ")

        lf = " ".join([x for x in lf if x not in self.useless_tokens]) # we are going to split again later!

        return [nl, lf, category]


def is_number(s:str):
    try:
        int(s)
        return True
    except ValueError:
        return False

@Preprocessor.register("simplified_cogs")
class COGSPreprocessor(Preprocessor):
    useless_tokens = {",", "_", "(", ")", "x"} # these tokens don't add anything semantically and could be reconstructed based on simple rules.

    def preprocess(self, row: List[str]) -> List[str]:

        nl, lf, category = row

        lf = lf.split(" ")

        copy_lf = []
        next_add_period = False
        for tok in lf:
            if tok == ".": # keep information but don't keep . as separate token as it might need to be copied otherwise.
                next_add_period = True
            elif tok in self.useless_tokens:
                pass
            elif is_number(tok): #removes variables --> information loss but for now it's fine.
                pass
            else:
                if next_add_period:
                    copy_lf.append("."+tok)
                else:
                    copy_lf.append(tok)
                next_add_period = False
        return [nl, " ".join(copy_lf), category]

@Preprocessor.register("aggressive_cogs")
class COGSSimplifiedAggressive(Preprocessor):
    """
    Aggressively prunes out tokens without a function.
    Simplification: removes variables!
    """
    useless_tokens = {",", "_", "(", ")", "x", ".", ";", "AND"}
    lambda_vars = {"a", "b", "e"}
    # the symbols a,b,e are variables in LAMBDA contexts.
    # we will first delete all occurrences of "." and then re-introduce them, since that's easier than selectively deleting them
    # (thanks to LAMBDA :/)
    lftok = {"agent", "theme", "recipient", "ccomp", "xcomp", "nmod", "in", "on", "beside"}

    def __init__(self, simplify: bool):
        self.simplify = simplify

    def preprocess(self, row: List[str]) -> List[str]:
        nl, lf, category = row

        lf = lf.split(" ")

        copy_lf = [tok for tok in lf if tok not in self.useless_tokens and not (self.simplify and (is_number(tok) or tok in self.lambda_vars))]
        copy_lf = [tok if not tok in self.lftok else "." + tok for tok in copy_lf]

        return [nl, " ".join(copy_lf), category]

    def _is_argument(self, str):
        return (is_number(str) or str in {"a", "b", "e"}) and str not in {"AND", ";", "*", "LAMBDA"} and str[0] != "."

    def postprocess(self, logical_form: List[str]) -> List[str]:
        """
        Reconstructs the original COGS logical form.
        :param logical_form:
        :return:
        """
        if self.simplify:
            raise ValueError("Cannot reconstruct original COGS logical form if variables are gone.")
        # Reinsert ;
        # pattern * [predicate] number --> * [predicate] number ;
        l = []
        for i in range(len(logical_form)):
            l.append(logical_form[i])
            if i >= 2 and is_number(logical_form[i]) and logical_form[i-2] == "*":
                l.append(";")

        logical_form = l
        l = []

        # insert AND
        for i in range(len(logical_form)):
            if not is_number(logical_form[i]) and logical_form[i] not in {".agent", ".theme", ".recipient", ".ccomp", ".xcomp", ";", "*", "a","b", "e", "."} \
                    and i > 0 and logical_form[i-1] not in {";", "*", "LAMBDA", "."} and logical_form[i][0].upper() != logical_form[i][0]:
                if i > 1 and logical_form[i-2] == "LAMBDA":
                    pass
                else:
                    l.append("AND")
            l.append(logical_form[i])

        logical_form = l
        l = []
        # Re-insert x _ and brackets
        i = 0
        while i < len(logical_form):
            if i < len(logical_form)-1 and self._is_argument(logical_form[i]) and self._is_argument(logical_form[i+1]):
                # give .agent 3 4 OR give .agent e a
                if logical_form[i] in {"a", "e", "b"}:
                    l.extend(["(", logical_form[i]])
                else:
                    l.extend(["(", "x", "_", logical_form[i]])

                if logical_form[i+1] in {"a", "e", "b"}:
                    l.extend([",", logical_form[i+1], ")"])
                else:
                    l.extend([",", "x", "_", logical_form[i+1], ")"])
                i += 1
            elif i < len(logical_form)-1 and self._is_argument(logical_form[i]) and logical_form[i+1][0].upper() == logical_form[i+1][0] \
                    and logical_form[i+1] not in {"AND", ";", "*", "LAMBDA"}:
            # give .agent 3 Emma
                l.extend(["(", "x", "_", logical_form[i], ",", logical_form[i+1], ")"])
                i += 1

            elif self._is_argument(logical_form[i]):
                if is_number(logical_form[i]):
                    l.extend(["(", "x", "_", logical_form[i], ")"])
                else:
                    l.extend(["(", logical_form[i], ")"])

            # Reinsert . after LAMBDA
            elif logical_form[i] == "LAMBDA" and i < len(logical_form)-1:
                l.extend(["LAMBDA", logical_form[i+1], "."])
                i += 1
            else:
                l.append(logical_form[i])

            i += 1

        # Expand .agent to . agent etc.
        logical_form = l
        l = []
        for i in range(len(logical_form)):
            if logical_form[i] in {".agent", ".theme", ".recipient", ".xcomp", ".ccomp", ".nmod", ".on", ".in", ".beside"}:
                l.extend([".", logical_form[i][1:]])
            else:
                l.append(logical_form[i])

        return l


@Preprocessor.register("cogs_google")
class COGSGooglePreprocessor(Preprocessor):
    useless_tokens = {",", "="} # these tokens don't add anything semantically and could be reconstructed based on simple rules.

    def preprocess(self, row: List[str]) -> List[str]:

        nl, lf, category = row

        nl = nl.strip(".").strip() # important: remove ., so the model is not forced to copy . in the outputs!

        lf = lf.split(" ")
        out_lf = []

        for i, tok in enumerate(lf):
            if tok in self.useless_tokens:
                pass
            elif tok == ".":
                out_lf[-1] = out_lf[-1] + tok
            else:
                out_lf.append(tok)

        lf = " ".join(out_lf) # we are going to split again later!

        return [nl, lf, category]


if __name__ == "__main__":
    p = COGSSimplifiedAggressive(False)
    with open("../../data/COGS/data/train.tsv") as f:
        for line in f:
            line = line.strip()
            nl, lf, cat = line.split("\t")
            _,lf_2, cat = p.preprocess([nl, lf, cat])

            if lf != " ".join(p.postprocess(lf_2.split(" "))):
                print("Original     ", lf)
                print("Reconstructed", " ".join(p.postprocess(lf_2.split(" "))))
                print("---")

