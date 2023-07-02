from typing import List

from nltk import Tree


from fertility.tree_utils.read_funql import get_tree, preorder_wo_brackets, \
    reconstruct_tree_without_brackets, QuoteHandling, tree2funql

GEO_ARITIES = {'stateid': 1, 'place': 1, 'riverid': 1, 'lowest': 1, 'area_1': 1, 'largest': 1, 'mountain': 1,
               'capital': 1, 'river': 1, 'shortest': 1, 'longer': 1, 'high_point_2': 1, 'elevation_1': 1,
               'traverse_1': 1, 'fewest': 1, 'most': 1, 'next_to_1': 1, 'loc_2': 1, 'lake': 1, 'state': 1,
               'traverse_2': 1, 'countryid': 1, 'exclude': 2, 'city': 1, 'high_point_1': 1, 'higher_2': 1, 'placeid': 1,
               'loc_1': 1, 'smallest_one': 1, 'elevation_2': 1, 'intersection': 2, 'size': 1, 'capital_2': 1,
               'capital_1': 1, 'sum': 1, 'highest': 1, 'count': 1, 'smallest': 1, 'cityid': 2, 'len': 1, 'density_1': 1,
               'major': 1, 'longest': 1, 'population_1': 1, 'next_to_2': 1, 'largest_one': 1, 'answer': 1,
               'low_point_2': 1, 'lower_2': 1, 'low_point_1': 1}

# Extracted from the KB
NAMED_ENTITIES = ['aberdeen', 'abilene', 'abingdon', 'ak', 'akron', 'al', 'alabama', 'alameda', 'alaska', 'albany',
                  'albuquerque', 'alexandria', 'alhambra', 'allegheny', 'allentown', 'altoona', 'alverstone',
                  'amarillo', 'anaheim', 'anchorage', 'anderson', 'annapolis', 'ann arbor', 'antero', 'appleton', 'ar',
                  'arizona', 'arkansas', 'arkansas river', 'arlington', 'arlington heights', 'arvada', 'atlanta',
                  'atlantic ocean', 'auburn', 'augusta', 'aurora', 'austin', 'az', 'backbone mountain', 'bakersfield',
                  'baltimore', 'bangor', 'baton rouge', 'bayonne', 'bear', 'beaumont', 'beaver dam creek', 'becharof',
                  'belford', 'belle fourche river', 'bellevue', 'bennington', 'berkeley', 'bethesda', 'bethlehem',
                  'bianca', 'bighorn', 'big stone lake', 'billings', 'biloxi', 'birmingham', 'bismarck', 'blackburn',
                  'black mesa', 'black mountain', 'bloomington', 'boise', 'bona', 'borah peak', 'boston', 'boulder',
                  'boundary peak', 'brasstown bald', 'bridgeport', 'bristol', 'bristol township', 'brockton',
                  'brookside', 'bross', 'browne tower', 'brownsville', 'buena park', 'buffalo', 'burbank', 'burlington',
                  'butte', 'ca', 'california', 'cambridge', 'camden', 'campbell hill', 'canadian', 'canton', 'carson',
                  'carson city', 'casper', 'castle', 'cedar rapids', 'centerville', 'champaign', 'champlain',
                  'charles mound', 'charleston', 'charlotte', 'chattahoochee', 'chattanooga', 'cheaha mountain',
                  'cheektowaga', 'cherry hill', 'chesapeake', 'cheyenne', 'chicago', 'chula vista', 'churchill',
                  'cicero', 'cimarron', 'cincinnati', 'citrus heights', 'clark fork', 'clearwater', 'cleveland',
                  'clifton', 'clingmans dome', 'clinton', 'co', 'colorado', 'colorado river', 'colorado springs',
                  'columbia', 'columbus', 'compton', 'concord', 'connecticut', 'corpus christi', 'costa mesa',
                  'covington', 'cranston', 'crestone', 'crestone needle', 'ct', 'cumberland', 'dakota', 'dallas',
                  'daly city', 'danbury', 'davenport', 'dayton', 'dc', 'de', 'dearborn', 'dearborn heights',
                  'death valley', 'decatur', 'delaware', 'delaware river', 'denver', 'des moines', 'detroit',
                  'district of columbia', 'dover', 'downey', 'driskill mountain', 'dubuque', 'duluth', 'dundalk',
                  'durham', 'duval circle', 'eagle mountain', 'east buttress', 'east los angeles', 'east orange',
                  'edison', 'elbert', 'el cajon', 'el diente', 'elgin', 'elizabeth', 'el monte', 'el paso', 'elyria',
                  'erie', 'escondido', 'essex', 'euclid', 'eugene', 'evans', 'evanston', 'evansville', 'ewa',
                  'fairbanks', 'fairfield', 'fairweather', 'fall river', 'fargo', 'farmington hills', 'fayetteville',
                  'fl', 'flathead', 'flint', 'florida', 'foraker', 'fort collins', 'fort lauderdale', 'fort smith',
                  'fort wayne', 'fort worth', 'framingham', 'frankfort', 'franklin township', 'fremont', 'fresno',
                  'fullerton', 'ga', 'gainesville', 'gannett peak', 'garden grove', 'garland', 'gary', 'georgetown',
                  'georgia', 'gila', 'glendale', 'grand forks', 'grand island', 'grand prairie', 'grand rapids',
                  'granite peak', 'grays', 'great falls', 'great salt lake', 'green', 'green bay', 'greensboro',
                  'greenville', 'greenwich', 'guadalupe peak', 'gulf of mexico', 'hamilton', 'hammond', 'hampton',
                  'harney peak', 'harrisburg', 'hartford', 'harvard', 'hattiesburg', 'hawaii', 'hayward', 'helena',
                  'hi', 'high point', 'hollywood', 'honolulu', 'houston', 'hubbard', 'hudson', 'humphreys peak',
                  'hunter', 'huntington', 'huntington beach', 'huntsville', 'huron', 'ia', 'id', 'idaho', 'idaho falls',
                  'il', 'iliamna', 'illinois', 'in', 'independence', 'indiana', 'indianapolis', 'inglewood', 'iowa',
                  'irondequoit', 'irvine', 'irving', 'irvington', 'jackson', 'jacksonville', 'jefferson city',
                  'jerimoth hill', 'jersey city', 'johnson township', 'joliet', 'juneau', 'kalamazoo', 'kansas',
                  'kansas city', 'kendall', 'kennedy', 'kenner', 'kenosha', 'kentucky', 'kettering', 'kings peak',
                  'kit carson', 'knoxville', 'koolaupoko', 'kootenai river', 'ks', 'ky', 'la', 'lafayette',
                  'lake champlain', 'lake charles', 'lake erie', 'lake michigan', 'lake of the woods', 'lake superior',
                  'lakewood', 'lansing', 'la plata', 'laramie', 'laredo', 'largo', 'las cruces', 'las vegas',
                  'lawrence', 'lawton', 'levittown', 'lewiston', 'lexington', 'lincoln', 'little missouri',
                  'little river', 'little rock', 'livonia', 'long beach', 'long island sound', 'longs', 'longview',
                  'lorain', 'los angeles', 'louisiana', 'louisville', 'lowell', 'lower merion', 'lubbock', 'lynchburg',
                  'lynn', 'ma', 'macon', 'madison', 'magazine mountain', 'maine', 'manchester', 'maroon', 'maryland',
                  'massachusetts', 'massive', 'mauna kea', 'mcallen', 'mckinley', 'md', 'me', 'medford', 'memphis',
                  'meriden', 'meridian', 'mesa', 'mesquite', 'metairie', 'mi', 'miami', 'miami beach', 'michigan',
                  'middletown', 'midland', 'mille lacs', 'milwaukee', 'minneapolis', 'minnesota', 'minot',
                  'mississippi', 'mississippi river', 'missoula', 'missouri', 'mn', 'mo', 'mobile', 'modesto', 'monroe',
                  'montana', 'montgomery', 'montpelier', 'mountain view', 'mount curwood', 'mount davis',
                  'mount elbert', 'mount frissell', 'mount greylock', 'mount hood', 'mount katahdin', 'mount mansfield',
                  'mount marcy', 'mount mckinley', 'mount mitchell', 'mount rainier', 'mount rogers', 'mount sunflower',
                  'mount vernon', 'mount washington', 'mount whitney', 'ms', 'mt', 'muncie', 'naknek', 'nashua',
                  'nashville', 'nc', 'nd', 'ne', 'nebraska', 'neosho', 'nevada', 'newark', 'new bedford', 'new britain',
                  'new hampshire', 'new haven', 'new jersey', 'new mexico', 'new orleans', 'newport beach',
                  'newport news', 'new rochelle', 'newton', 'new york', 'nh', 'niagara falls', 'niobrara', 'nj', 'nm',
                  'norfolk', 'norman', 'north carolina', 'north charleston', 'north dakota', 'north little rock',
                  'north palisade', 'north platte', 'norwalk', 'nv', 'ny', 'oakland', 'oak lawn', 'oceanside',
                  'ocheyedan mound', 'odessa', 'ogden', 'oh', 'ohio', 'ohio river', 'ok', 'okeechobee', 'oklahoma',
                  'oklahoma city', 'olympia', 'omaha', 'ontario', 'or', 'orange', 'oregon', 'orlando', 'ouachita',
                  'ouachita river', 'overland park', 'owensboro', 'oxnard', 'pa', 'pacific ocean', 'parkersburg',
                  'parma', 'pasadena', 'paterson', 'pawtucket', 'pearl', 'pecos', 'penn hills', 'pennsylvania',
                  'pensacola', 'peoria', 'philadelphia', 'phoenix', 'pierre', 'pine bluff', 'pittsburgh', 'plano',
                  'pocatello', 'pomona', 'pontchartrain', 'pontiac', 'port arthur', 'portland', 'portsmouth', 'potomac',
                  'potomac river', 'powder', 'princeton', 'providence', 'provo', 'pueblo', 'quandary', 'quincy',
                  'racine', 'rainier', 'rainy', 'raleigh', 'rapid city', 'reading', 'red', 'red bluff reservoir',
                  'redford', 'redondo beach', 'red river', 'reno', 'republican', 'rhode island', 'ri', 'richardson',
                  'richmond', 'rio grande', 'riverside', 'roanoke', 'rochester', 'rock', 'rockford', 'rock springs',
                  'roswell', 'royal oak', 'rutland', 'sacramento', 'saginaw', 'salem', 'salinas', 'salt lake city',
                  'salton sea', 'san angelo', 'san antonio', 'san bernardino', 'san diego', 'sanford', 'san francisco',
                  'san jose', 'san juan', 'san leandro', 'san mateo', 'santa ana', 'santa barbara', 'santa clara',
                  'santa fe', 'santa monica', 'santa rosa', 'sassafras mountain', 'savannah', 'sc', 'schenectady',
                  'scottsdale', 'scotts valley', 'scranton', 'sd', 'seattle', 'shasta', 'shavano', 'shreveport', 'sill',
                  'silver spring', 'simi valley', 'sioux city', 'sioux falls', 'sitka', 'skokie', 'smoky hill', 'snake',
                  'snake river', 'somerville', 'south bend', 'south buttress', 'south carolina', 'south dakota',
                  'southeast corner', 'southfield', 'south gate', 'south platte', 'sparks', 'spokane', 'springfield',
                  'spruce knob', 'stamford', 'st. clair', 'st. clair shores', 'st. elias', 'sterling heights',
                  'st. francis', 'st. francis river', 'st. joseph', 'st. louis', 'stockton', 'st. paul',
                  'st. petersburg', 'sunnyvale', 'sunrise manor', 'superior', 'syracuse', 'tacoma', 'tahoe',
                  'tallahassee', 'tampa', 'taum sauk mountain', 'taylor', 'tempe', 'tenleytown', 'tennessee',
                  'terre haute', 'teshekpuk', 'texas', 'thousand oaks', 'timms hill', 'tn', 'toledo', 'tombigbee',
                  'topeka', 'torrance', 'torreys', 'trenton', 'troy', 'tucson', 'tulsa', 'tuscaloosa', 'tx', 'tyler',
                  'uncompahgre', 'upper darby', 'ut', 'utah', 'utica', 'va', 'vallejo', 'vancouver', 'ventura',
                  'verdigris river', 'vermont', 'virginia', 'virginia beach', 'vt', 'wa', 'wabash', 'waco', 'wahiawa',
                  'waltham', 'walton county', 'warren', 'warwick', 'washington', 'washita', 'waterbury',
                  'wateree catawba', 'waterford', 'waterloo', 'watertown', 'waukegan', 'west allis', 'west covina',
                  'west hartford', 'westland', 'westminster', 'west palm beach', 'west valley', 'west virginia',
                  'wheeler peak', 'wheeling', 'white', 'white butte', 'whitney', 'whittier', 'wi', 'wichita',
                  'wichita falls', 'williamson', 'wilmington', 'wilson', 'winnebago', 'winston-salem', 'wisconsin',
                  'woodall mountain', 'woodbridge', 'worcester', 'wrangell', 'wv', 'wy', 'wyoming', 'yale',
                  'yellowstone', 'yonkers', 'youngstown']

NAMED_ENTITY_TRIE = dict()
for n in NAMED_ENTITIES:
    parts = n.split(" ")
    for i in range(len(parts)):
        path = tuple(parts[:i])
        NAMED_ENTITY_TRIE[path] = NAMED_ENTITY_TRIE.get(path, set()) | {tuple(parts[:i+1])}

GEO_PREDS = set(GEO_ARITIES.keys())

def get_longest_match(toks, trie):
    """
    If there is a match at the beginning of toks, it returns the last index where it matches.
    It doesn't backtrack if there is a continuation that turns out to be a dead end.
    Looks OK for our purpose.
    :param toks:
    :param trie:
    :return:
    """
    state = ()
    index = -1
    for i, tok in enumerate(toks):
        next_state = state + (tok,)
        if state not in trie:
            break
        if next_state in trie[state]:
            state = next_state
            index += 1
        else:
            break
    if index < 0 or state in trie:
        return None
    return index


# print(get_longest_match(["new", "york"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["montana"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["dfgdfg"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["new", "york", "bla"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["new", "york", "bla", "blub"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["mountain", "view"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["mountain", "blub"], NAMED_ENTITY_TRIE))
# print(get_longest_match(["mountain"], NAMED_ENTITY_TRIE))


# print(NAMED_ENTITY_TRIE)

def restore_quotes(s: List[str]):
    r = []
    i = 0
    while i < len(s):
        match = get_longest_match(s[i:], NAMED_ENTITY_TRIE)  # O(n^2) but whatever
        if match is not None:
            r.append("'" + " ".join(s[i:i+match+1]) + "'")
            i += match+1
        else:
            r.append(s[i])
            i += 1
    return r


# def rectify_quotes(t: Tree) -> Tree:
#     if isinstance(t, str):
#         return t
#     children = list(t)
#

# def get_geo_tree(s: str):
#     t = get_tree(s)
#     # if we have a named entity like "new york" that is split over multiple tokens,
#     # there will be a node that has multiple children



def geo_reconstruct_tree(s: List[str], add_quotes: bool):
    if add_quotes:
        s = restore_quotes(s)
    return reconstruct_tree_without_brackets(s, GEO_ARITIES)


if __name__ == "__main__":
    import json
    from fertility.eval.geo_eval.executor_geo import ProgramExecutorGeo
    executor = ProgramExecutorGeo()
    # Weird empty denotation for the following examples (also the case for Herzig and Berant...):
    # answer(elevation_1(placeid('death valley')))
    # answer(size(city(cityid('new york',_))))

    with open("data/span_based_herzig_and_berant/geo/funql/test.json") as f:
        # f = ["{\"program\": \"answer ( elevation_1 ( highest ( mountain ( loc_2 ( stateid ( 'texas' ) ) ) ) ) )\"}"]
        for line in f:
            j = json.loads(line)
            program = j["program"]
            t = get_tree(program)
            toks = preorder_wo_brackets(t, QuoteHandling.DELETE)
            print(toks)
            #Reconstruct tree
            t = geo_reconstruct_tree(toks, True)
            funql_rep = tree2funql(t)
            gold_answer = executor.execute(program)
            reconstructed_answer = executor.execute(funql_rep)

            if gold_answer != reconstructed_answer:
                print("Error")
                print("input", program)
                print("reconstrcted", t, funql_rep)
                print("Gold answer", gold_answer)
                print("answer for reconstructed", reconstructed_answer)
                input()


    mrs = ["answer(population_1(cityid('washington','dc')))",
           "answer ( population_1 ( cityid ( 'washington', 'dc' ) ) )",
           "answer ( population_1 ( largest ( city ( loc_2 ( state ( stateid ( 'new york' ) ) ) ) ) ) )"]

    # for mr in mrs:
    #     print(mr)
    #     t = get_tree(mr)
    #     toks = preorder_wo_brackets(t, QuoteHandling.DELETE)
    #     print(toks)
    #
    #     print("restored quotes", restore_quotes(toks))
    #
    #     print(t)
    #     funql_rep = tree2funql(t)
    #     print("funql:", funql_rep)
    #     # t2 = reconstruct_tree_with_quotes(toks, GEO_ARITIES)
    #     # t2 = reconstructed = geo_reconstruct(toks, "'" if remove_quotes else "")
    #     # print("Orig tree")
    #     # t.pretty_print()
    #     # print("Reonconstructed")
    #     # t2.pretty_print()
    #     print("----")
