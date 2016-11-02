import traceback

from submissions.Ottenlips import billionaires

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

bill = DataFrame()


list_of_billionaire = billionaires.get_billionaires()

def billTarget(string):
    if(billionaires['demographics']['age']<30 and billionaires['demographics']['age'] != -1):
        return 1
    else:
        return 0

for billionaires in list_of_billionaire:
    # print(billionaires['wealth']['type'])
    #print(billionaires)
    bill.target.append(billTarget(billionaires['demographics']['age']))
    # bill.target.append(billionaires['wealth']['how']['inherited'])
    bill.data.append([

                    float(billionaires['location']['gdp']),
                    float(billionaires['wealth']['worth in billions']),
           billionaires['rank'],
           # billionaires['demographics']['age'],

    ])


bill.feature_names = [
    'gdp',
    'worth',
    'rank',
    'age',
]

bill.target_names = [
   'old',
    'young',
]

Examples = {
    'Billionaires': bill,
}