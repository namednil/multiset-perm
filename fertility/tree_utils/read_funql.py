import re
from enum import Enum
from typing import List, Dict, Union, Set, Tuple

from nltk import Tree, ImmutableTree

s = re.compile("([,() ])")

quotes = re.compile("(')")


def split_tokens(x):
    return [x for x in s.split(x.strip()) if x and x.strip(" ")]


def to_lisp(toks):
    r = []
    for t in toks:
        if t == "(":
            r = r[:-1] + ["("] + [r[-1]]
        elif t == ",":
            r.append(" ")
        else:
            r.append(t)
    return r


def get_tree(s):
    s = split_tokens(s)
    return Tree.fromstring(" ".join(to_lisp(s)))


def tree2funql(t):
    if isinstance(t, str):
        return t
    return t.label() + "(" + ",".join(tree2funql(c) for c in t) + ")"


# def preorder_with_brackets(t, ambiguous_arities):
#     children = list(t)
#     crs = [preorder_with_brackets(c, ambiguous_arities) if isinstance(c, Tree) else c for c in children]
#     if len(crs) == 0:
#         return t
#     if t.label() in ambiguous_arities:
#         return "( " + t.label() + " " + " ".join(crs) + " )"
#     else:
#         return t.label() + " " + " ".join(crs)


class QuoteHandling(Enum):
    NOOP = 0
    SEPARATE = 1
    DELETE = 2


def preorder_wo_brackets(t: Union[str, Tree], quote_handling: QuoteHandling) -> List[str]:
    if isinstance(t, str):
        if quote_handling == QuoteHandling.SEPARATE:
            return [x for x in quotes.split(t) if x]
        elif quote_handling == QuoteHandling.DELETE:
            return [t.replace("'", "")]
        else:
            return [t]
    children = list(t)
    r = [t.label()]
    for c in children:
        r.extend(preorder_wo_brackets(c, quote_handling))
    return r


def reconstruct_tree_without_brackets(s: List[str], arities: Dict[str, int]) -> Tree:
    """
    Reconstruct a tree from a linearized representation without brackets.
    :param s:
    :param arities:
    :param add_quotes:
    :param joiner:
    :return:
    """
    position = 0

    def read_tree():
        nonlocal position
        head = s[position]
        position += 1
        if head in arities:
            return Tree(head, children=[read_tree() for _ in range(arities[head])])
        else:
            return head  # assume arity 0

    t = read_tree()
    assert position == len(s)
    # ~ print(position, len(s))
    return t


def reconstruct_tree_with_partial_brackets(s: List[str], arities: Dict[str, int]) -> Tree:
    position = 0

    def read_tree():
        nonlocal position
        head = s[position]
        position += 1
        if head == "(":
            node_name = s[position]
            position += 1
            trees = []
            while s[position] != ")":
                trees.append(read_tree())
            position += 1
            return Tree(node_name, children=trees)

        elif head in arities:
            return Tree(head, children=[read_tree() for _ in range(arities[head])])
        else:
            return head

    t = read_tree()
    assert position == len(s)
    return t


def join_quotes(s: List[str]) -> List[str]:
    """
    Compresses a sequence of unknown tokens into a single element (a tuple).
    :param s:
    :param known:
    :return:
    """
    last_part = ""
    r = []
    in_quotes = False
    for i in range(len(s)):
        if in_quotes:
            last_part += " " + s[i]
            if s[i].endswith("'"):
                in_quotes = False
                r.append(last_part)
                last_part = ()
        else:

            if s[i].startswith("'") == s[i].endswith("'"):  # if no quotes or quotes on both sides
                r.append(s[i])
            elif s[i].startswith("'"):
                last_part = s[i]
                in_quotes = True

    if last_part:
        r.append(last_part)

    return r


def sort_tree_by_hash(t: ImmutableTree, sortable_nodes: Set[str]) -> ImmutableTree:
    if not isinstance(t, ImmutableTree) and not isinstance(t, str):
        raise ValueError("Got an unexpected type!")

    if isinstance(t, str):
        return t

    if t.label() in sortable_nodes:
        return ImmutableTree(t.label(),
                             children=sorted((sort_tree_by_hash(c, sortable_nodes) for c in t), key=lambda x: hash(x)))
    else:
        return ImmutableTree(t.label(), children=[sort_tree_by_hash(c, sortable_nodes) for c in t])


if __name__ == "__main__":
    # x = "answer _stop_1 _flight ( intersection _airline_2 airline_code dl _flight_number_2 flight_number 838 _from_2 city_name san_francisco _to_2 city_name atlanta )"
    x = "answer _stop_1 _flight ( intersection _airline_2 airline_code dl ( intersection _airline_2 airline_code dl _airline_2 airline_code BLA ) _flight_number_2 flight_number 838 _from_2 city_name san_francisco _to_2 city_name atlanta )"
    x2 = "answer _stop_1 _flight ( intersection _from_2 city_name san_francisco _airline_2 airline_code dl ( intersection _airline_2 airline_code BLA _airline_2 airline_code dl ) _flight_number_2 flight_number 838 _to_2 city_name atlanta )"
    x = x.split(" ")
    x2 = x2.split(" ")
    arities = {"answer": 1, "_stop_1": 1,
               "_flight": 1,"_airline_2": 1, "airline_code": 1,
               "_flight_number_2": 1,"flight_number": 1, "_from_2": 1,"city_name": 1, "_to_2": 1
               }
    t = reconstruct_tree_with_partial_brackets(x, arities)
    t2 = reconstruct_tree_with_partial_brackets(x2, arities)

    t_sort = sort_tree_by_hash(ImmutableTree.convert(t), {"intersection"})
    t2_sort = sort_tree_by_hash(ImmutableTree.convert(t2), {"intersection"})

    print(t_sort)
    print(t2_sort)
    print(t_sort == t2_sort)
