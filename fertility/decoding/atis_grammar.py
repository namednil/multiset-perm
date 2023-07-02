from typing import Dict

import numpy as np
from nltk import CFG, Production, Nonterminal

from fertility.decoding.cnf_helper import to_cnf, compute_derivable_lengths
from fertility.decoding.decoding_grammar import DecodingGrammar, DecodingGrammarFast

# The following grammar is essentially that of Guo et al. 2020, "Benchmarking Meaning Representations in Neural Semantic Parsing"
# https://github.com/JasperGuo/Unimer/blob/master/grammars/atis/typed_funql_grammar.py

# Differences: Removed all brackets except those for "or" and "intersection" where they are necessary
# because these operators don't have fixed arity.

# The handling of intersection and union is not perfect though because we can switch types (intersect city with aircraft)

ATIS_BRACKET_GRAMMAR = CFG.fromstring("""
Query -> "answer" Predicate 
Predicate -> Flight | Time | Airline | Airport | Aircraft | Stop | FlightNumber | Fare | City | GroundTransport | TimeZone | Meta | other | BookingClass | Meal
Time -> "_arrival_time" Flight  | "_departure_time" Flight  | "_stop_arrival_time" Flight  | "_time_elapsed" Flight  | "_minimum_connection_time" Airport  | "_minutes_distant" GroundTransport  | "days_code" "sa"
Stop -> "_stops" Flight 
FlightNumber -> "_flight_number" Flight  | "flight_number" flight_number_value 
Flight -> "_airline_2" Airline  | "_aircraft_2" Aircraft  | "_flight"      "all"    | "_connecting" Flight  | "_discounted" Flight  | "_economy" Flight  | "_flight_number_2" FlightNumber  | "_flight" Flight  | "_from_2" City  | "_from_2" Airport  | "_has_meal" Flight  | "_has_stops" Flight  | "_nonstop" Flight  | "_oneway" Flight  | "_round_trip" Flight  | "_round_trip" Flight  | "_to_2" City  | "_to_2" Airport  | "_to_2" state_name  | "_jet" Flight  | "_turboprop" Flight   | "_day_2" day  | "_after_day_2" day  | "_before_day_2" day  | "_day_arrival_2" day  | "_approx_arrival_time_2" time  | "_approx_return_time_2" time  | "_approx_departure_time_2" time  | "_arrival_time_2" time  | "_departure_time_2" time  | "_daily" Flight  | "_day_after_tomorrow" Flight  | "_day_number_2" day_number  | "_day_number_arrival_2" day_number  | "_day_number_return_2" day_number  | "_day_return_2" day  | "_days_from_today_2" integer  | "_during_day_2" day_period  | "_during_day_arrival_2" day_period  | "_month_2" month  | "_month_arrival_2" month  | "_month_return_2" month  | "_next_days_2" integer  | "_overnight" Flight  | "_today" Flight  | "_tomorrow" Flight  | "_tomorrow_arrival" Flight  | "_tonight" Flight  | "_weekday" Flight  | "_year_2" year  | "_fare_2" dollar  | "_fare_basis_code_2" Fare  | "_stop_2" City  | "_stop_2" Airport  | "_stops_2" integer  | "_booking_class_2" class_description  | "_class_type_2" class_description  | "_meal_2" Meal  | "_meal_code_2" Meal  | "_<_arrival_time_2" time  | "_<_departure_time_2" time  | "_<_fare_2" dollar  | "_>_arrival_time_2" time  | "_>_departure_time_2" time  | "_>_departure_time_2" Flight  | "_>_capacity_2" Aircraft  | "_>_stops_2" integer  | "argmax_arrival_time" Flight  | "argmax_departure_time" Flight  | "argmax_fare" Flight  | "argmax_stops" Flight  | "argmin_capacity" Flight  | "argmin_arrival_time" Flight  | "argmin_departure_time" Flight  | "argmin_fare" Flight  | "argmin_stops" Flight  | "argmin_time_elapsed" Flight 
Flight -> "(" "intersection" Flight Conjunction ")"  | "(" "or" Flight Conjunction ")" | "not"    Flight 
Flight -> "_time_elapsed_2" "hour" integer_value |  '_manufacturer_2' 'manufacturer' 'boeing'
Conjunction -> Predicate Conjunction | Predicate
Fare -> "_fare" Flight  | "_fare_basis_code" Flight  | "_fare_basis_code"      "all"    | "fare_basis_code" fare_basis_code_value  | "_ground_fare" GroundTransport 
Fare -> "(" "or"      Fare    Fare ")"
Airline -> "_airline" Airline  | "_abbrev" Airline  | "_airline_1" Flight  | "_airline_name" Flight  | "airline_code" airline_code_value  | "_services_2" City  | "_services_2" Airport  | "argmax_count" Airline  | "_airline"      "all"  
Airline -> "(" "intersection"      Airline    Conjunction ")"  | "(" "or"      Airline    Conjunction ")"  | "not"    Airline 
Aircraft -> "argmin_capacity" Aircraft  | "argmax_capacity" Aircraft  | "aircraft_code" aircraft_code_value  | "_aircraft_1" Flight  | "_airline_2" Airline  | "_aircraft_basis_type_2" basis_type  | "_aircraft" Aircraft  | "_jet" Aircraft  | "_turboprop" Aircraft  | "_manufacturer_2" "manufacturer" "boeing" | "_aircraft"      "all"   
Aircraft -> "(" "intersection"      Aircraft    Conjunction ")"  | "(" "or"      Aircraft    Conjunction ")"  | "not"    Aircraft 
Airport -> "_airport" Airport  | "_airport_1" City  | "_from_1" Flight  | "_to_1" Flight  | "_stop_1" Flight  | "airport_code" airport_code_value  | "_loc:_t_2" City  | "_loc:_t_2" state_name  | "argmin_miles_distant_2" Airport  | "_airport"      "all"    
Airport -> "(" "intersection"      Airport    Conjunction ")"  | "(" "or"      Airport    Conjunction ")"  | "not"    Airport  | "_services_1" airline_code 
City -> "_city" City  | "city_name" city_name_value  | "_city"      "all"    | "_to_1" Flight  | "_from_1" Flight  | "_services_1" airline_code  | "_loc:_t_1" Airport 
City -> "(" "intersection"      City    Conjunction ")"
BookingClass -> "_booking_class_1" Flight  | "_booking_class:_t"      "all"    | "_class_of_service"      "all"   
GroundTransport -> "_air_taxi_operation" GroundTransport  | "_limousine" GroundTransport  | "_rapid_transit" GroundTransport  | "_rental_car" GroundTransport  | "_from_airport_2" Airport  | "_from_airport_2" City  | "_ground_transport" GroundTransport  | "_to_city_2" City  | "_taxi" GroundTransport  | "_ground_transport"      "all"    
GroundTransport -> "(" "intersection"      GroundTransport    Conjunction ")" | "(" "or"      GroundTransport    Conjunction ")" | "_weekday" GroundTransport 
Meal -> "_meal" Flight  | "_meal_code"      "all"    | "meal_code" meal_code_value  | "meal_description" meal_description_value 
Meal -> "(" "intersection"      Meal    Conjunction ")"
TimeZone -> "_time_zone_code" TimeZone  | "_loc:_t_1" City 
Meta -> "_equals"   Airline   Airline    | "_equals"   Airport    Airport    | "_max" Fare  | "_min" Fare  | "count" Flight  | "count" Airport  | "count" Airline  | "count" BookingClass  | "count" Fare  | "count" City  | "sum_capacity" Aircraft  | "sum_stops" Flight 
other -> "_services" Airline City  | "_capacity" Aircraft  | "_capacity" Flight  | "_restriction_code" Flight  | "_flight_airline" Flight  | "_flight_fare" Flight  | "_flight_aircraft" Flight  | "_fare_time" Flight  | "_miles_distant"   Airport      City    | "_named_1" Airline 


city_name_value -> "cleveland" | "milwaukee" | "detroit" | "los_angeles" | "miami" | "salt_lake_city" | "ontario" | "tacoma" | "memphis" | "denver" | "san_francisco" | "new_york" | "tampa" | "washington" | "westchester_county" | "boston" | "newark" | "pittsburgh" | "charlotte" | "columbus" | "atlanta" | "oakland" | "kansas_city" | "st_louis" | "nashville" | "chicago" | "fort_worth" | "san_jose" | "dallas" | "philadelphia" | "st_petersburg" | "baltimore" | "san_diego" | "cincinnati" | "long_beach" | "phoenix" | "indianapolis" | "burbank" | "montreal" | "seattle" | "st_paul" | "minneapolis" | "houston" | "orlando" | "toronto" | "las_vegas"
basis_type -> "basis_type" basis_type_value 
basis_type_value -> "737" | "767"
flight_number_object -> "flight_number" flight_number_value 
flight_number_value -> "1291" | "345" | "813" | "71" | "1059" | "212" | "1209" | "281" | "201" | "324" | "19" | "352" | "137338" | "4400" | "323" | "505" | "825" | "82" | "279" | "1055" | "296" | "315" | "1765" | "405" | "771" | "106" | "2153" | "257" | "402" | "343" | "98" | "1039" | "217" | "539" | "459" | "417" | "1083" | "3357" | "311" | "210" | "139" | "852" | "838" | "415" | "3724" | "21" | "928" | "269" | "270" | "297" | "746" | "1222" | "271"
day -> "day" day_value 
day_value -> "monday" | "wednesday" | "thursday" | "tuesday" | "saturday" | "friday" | "sunday"
time -> "time" time_value 
time_value -> "1850" | "1110" | "2000" | "1815" | "1024" | "1500" | "1900" | "1600" | "1300" | "1800" | "1200" | "1628" | "1830" | "823" | "1245" | "1524" | "200" | "1615" | "1230" | "705" | "1045" | "1700" | "1115" | "1645" | "1730" | "815" | "0" | "500" | "1205" | "1940" | "1400" | "1130" | "2200" | "645" | "718" | "2220" | "600" | "630" | "800" | "838" | "1330" | "845" | "1630" | "1715" | "2010" | "1000" | "1619" | "2100" | "1505" | "2400" | "1923" | "100" | "1145" | "2300" | "1620" | "2023" | "2358" | "1425" | "720" | "1310" | "700" | "650" | "1410" | "1030" | "1900" | "1017" | "1430" | "900" | "1930" | "1133" | "1220" | "2226" | "1100" | "819" | "755" | "2134" | "555" | "1"
day_number -> "day_number" day_number_value 
day_number_value -> "13" | "29" | "28" | "22" | "21" | "16" | "30" | "12" | "18" | "19" | "31" | "20" | "27" | "6" | "26" | "17" | "11" | "10" | "15" | "23" | "24" | "25" | "14" | "1" | "3" | "8" | "5" | "2" | "9" | "4" | "7"
integer -> "integer" integer_value 
integer_value -> "2" | "1" | "3" | "9"
day_period -> "day_period" day_period_value 
day_period_value -> "early" | "afternoon" | "late_evening" | "late_night" | "mealtime" | "evening" | "pm" | "daytime" | "breakfast" | "morning" | "late"
month -> "month" month_value 
month_value -> "april" | "august" | "may" | "october" | "june" | "november" | "september" | "february" | "december" | "march" | "july" | "january"
dollar -> "dollar" dollar_value 
dollar_value -> "1000" | "1500" | "466" | "1288" | "300" | "329" | "416" | "124" | "932" | "1100" | "200" | "500" | "100" | "415" | "150" | "400"
fare_basis_code_value -> "qx" | "qw" | "qo" | "fn" | "yn" | "bh" | "k" | "b" | "h" | "f" | "q" | "c" | "y" | "m"
class_description -> "class_description" class_description_value 
class_description_value -> "thrift" | "coach" | "first" | "business"
meal_description_value -> "snack" | "breakfast" | "lunch" | "dinner"
meal_code_value -> "ap_58" | "ap_57" | "d_s" | "b" | "ap_55" | "s_" | "sd_d" | "ls" | "ap_68" | "ap_80" | "ap" | "s"
airline_code -> "airline_code" airline_code_value 
airline_code_value -> "usair" | "co" | "ua" | "delta" | "as" | "ff" | "canadian_airlines_international" | "us" | "nx" | "hp" | "aa" | "kw" | "ml" | "nw" | "ac" | "tw" | "yx" | "ea" | "dl" | "wn" | "lh" | "cp"
airport_code_value -> "dallas" | "ont" | "stapelton" | "bna" | "bwi" | "iad" | "sfo" | "phl" | "pit" | "slc" | "phx" | "lax" | "bur" | "ind" | "iah" | "dtw" | "las" | "dal" | "den" | "atl" | "ewr" | "bos" | "tpa" | "jfk" | "mke" | "oak" | "yyz" | "dfw" | "cvg" | "hou" | "lga" | "ord" | "mia" | "mco"
year -> "year" year_value 
year_value -> "1991" | "1993" | "1992"
aircraft_code_value -> "m80" | "dc10" | "727" | "d9s" | "f28" | "j31" | "767" | "734" | "73s" | "747" | "737" | "733" | "d10" | "100" | "757" | "72s"
state_name -> "state_name" state_name_value 
state_name_value -> "minnesota" | "florida" | "arizona" | "nevada" | "california"
""")

ATIS_BRACKET_GRAMMAR = to_cnf(ATIS_BRACKET_GRAMMAR)

@DecodingGrammar.register("atis_bracket_slow")
class AtisBracketDecodingGrammar(DecodingGrammar):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(ATIS_BRACKET_GRAMMAR)
        # This is just to save time, only strings of length 0,1,2 are not derivable by the grammar.
        self.derivable_lengths = np.array([False, False, False] + 1000*[True])


@DecodingGrammar.register("atis_bracket")
class AtisBracketDecodingGrammar(DecodingGrammarFast):

    def __init__(self, tok2id: Dict[str, int]):
        super().__init__(tok2id)
        self.set_grammar(ATIS_BRACKET_GRAMMAR)
        # This is just to save time, only strings of length 0,1,2 are not derivable by the grammar.
        self.derivable_lengths = np.array([False, False, False] + 1000*[True])


if __name__ == "__main__":
    from nltk.parse.chart import BottomUpChartParser

    def recognize(parser, s, start_symbol):
        chart = parser.chart_parse(s)
        it = chart.parses(start_symbol)
        try:
            t = next(it)
            return t
        except StopIteration:
            return False

    # print(GEO_BASE_GRAMMAR)

    parser = BottomUpChartParser(ATIS_BRACKET_GRAMMAR)
    problems = 0
    with open("data/atis/atis_funql_train_brackets.tsv") as f:
        for line in f:
            nl, mr = line.strip().split("\t")
            mr = mr.split(" ")
            if not recognize(parser, mr, ATIS_BRACKET_GRAMMAR.start()):
                print("Can't parse:")
                print(nl, mr)
                problems += 1
    print(problems)