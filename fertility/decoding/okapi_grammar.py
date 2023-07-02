from typing import Dict

import numpy as np
from nltk import CFG, Production, Nonterminal

from fertility.decoding.cnf_helper import to_cnf
from fertility.decoding.decoding_grammar import DecodingGrammar, DecodingGrammarFast


def add_numbers(g: CFG):
    new_productions = g.productions()
    for i in range(100):
        new_productions.append(Production(Nonterminal("Number"), [str(i)]))

    return CFG(g.start(), new_productions)

OKAPI_EMAIL = CFG.fromstring("""
S -> Clause | Clause S
Clause -> "GET" "message"
Clause -> "FILTER" StringProperty StringCond StringValue
Clause -> "FILTER" TimeProperty StringCond TimeValue
Clause -> "FILTER" BoolProperty BoolCond BoolValue
Clause -> "ORDERBY" TimeProperty Order
Clause -> "SEARCH" StringValue
Clause -> "TOP" Number
Clause -> "COUNT"
Order -> "asc" | "desc"
BoolValue -> "True" | "False"
BoolCond -> "eq"
StringCond -> "eq"
BoolProperty -> 'message.hasAttachments' | 'message.isRead'
StringProperty ->  'message.categories' | 'message.importance' | "message.from"
TimeProperty -> "message.receivedDateTime" | "message.sentDateTime"
StringValue -> "@COPY@" | "@COPY@" StringValue | Number
TimeValue -> "today" | "@COPY@" | "@COPY@" TimeValue
Number -> "@COPY@"
""")


OKAPI_EMAIL = add_numbers(OKAPI_EMAIL)
OKAPI_EMAIL = to_cnf(OKAPI_EMAIL)


OKAPI_CALENDAR = CFG.fromstring("""
S -> Clause | Clause S
Clause -> "GET" "event" | "GET" "@COPY@"
Clause -> "FILTER" StringProperty StringCond StringValue
Clause -> "FILTER" TimeProperty TimeCond TimeValue
Clause -> "FILTER" BoolProperty BoolCond BoolValue
Clause -> "ORDERBY" TimeProperty Order
Clause -> "SEARCH" StringValue
Clause -> "TOP" Number
Clause -> "COUNT"
Order -> "asc" | "desc"
BoolValue -> "True" | "False"
BoolCond -> "eq"
StringCond -> "eq"
TimeCond -> "eq" | "lt" | "gt"
BoolProperty -> "event.hasAttachments"
StringProperty ->  "event.calendar" | "event.categories" | "event.importance" | "event.location" | "event.organizer"
TimeProperty -> "event.start" | "event.end"
StringValue -> "@COPY@" | "@COPY@" StringValue | Number
StringValue -> "High" | "Low" | "OtherAppointments"
TimeValue -> "today" | "@COPY@" | "@COPY@" TimeValue
Number -> "@COPY@"
""")

OKAPI_CALENDAR = add_numbers(OKAPI_CALENDAR)
OKAPI_CALENDAR = to_cnf(OKAPI_CALENDAR)

OKAPI_DOCUMENT = CFG.fromstring("""
S -> Clause | Clause S
Clause -> "GET" "drive.recent" | "GET" "drive.root.children" | "GET" "drive.sharedWithMe"
Clause -> "FILTER" StringProperty StringCond StringValue
Clause -> "FILTER" TimeProperty TimeCond TimeValue
Clause -> "FILTER" BoolProperty BoolCond BoolValue
Clause -> "ORDERBY" TimeProperty Order
Clause -> "SEARCH" StringValue
Clause -> "TOP" Number
Clause -> "COUNT"
Order -> "asc" | "desc"
BoolValue -> "True" | "False"
BoolCond -> "eq"
StringCond -> "eq"
TimeCond -> "eq" | "lt" | "gt"
StringProperty ->  "file.createdBy" | "file.fileType" | "file.lastModifiedBy" | "file.sharedWith"
TimeProperty -> "file.createdDateTime" | "file.lastModifiedDateTime" | "file.size" | "file.name" 
StringValue -> "@COPY@" | "@COPY@" StringValue | Number
StringValue -> "doc" | "xlsx" | "png" | "jpeg" | "ppt"
TimeValue -> "today" | "@COPY@" | "@COPY@" TimeValue
Number -> "@COPY@"
""")

OKAPI_DOCUMENT = add_numbers(OKAPI_DOCUMENT)
OKAPI_DOCUMENT = to_cnf(OKAPI_DOCUMENT)


@DecodingGrammar.register("okapi_email")
class OkapiEmailGrammar(DecodingGrammarFast):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(OKAPI_EMAIL)
        # To save a little time:
        self.derivable_lengths = np.array([False] + 1000*[True])


@DecodingGrammar.register("okapi_calendar")
class OkapiCalendarGrammar(DecodingGrammarFast):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(OKAPI_CALENDAR)
        # To save a little time:
        self.derivable_lengths = np.array([False] + 1000*[True])


@DecodingGrammar.register("okapi_document")
class OkapiDocumentGrammar(DecodingGrammarFast):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(OKAPI_DOCUMENT)
        # To save a little time:
        self.derivable_lengths = np.array([False] + 1000*[True])


