from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import traceback
from submissions.Ottenlips import billionaires


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
    elif(billionaires['demographics']['age'] != -1):
        return 0
    else:
        return 2

for billionaires in list_of_billionaire:
    # print(billionaires['wealth']['type'])
    #print(billionaires)
    bill.target.append(billTarget(billionaires['demographics']['age']))
    # bill.target.append(billionaires['wealth']['how']['inherited'])
    bill.data.append([

        float(billionaires['location']['gdp']),
        float(billionaires['wealth']['worth in billions']),
        float(billionaires['rank']),
           # billionaires['demographics']['age'],

    ])


bill.feature_names = [
    'gdp of origin country',
    'worth',
    'rank',
    # 'age',
]

bill.target_names = [
    'old',
    'young',
    'age not listed'
]



'''
Make a customn classifier,
'''
mlpc = MLPClassifier(
    # hidden_layer_sizes = (100,),
    # activation = 'relu',
    solver='sgd', # 'adam',
    # alpha = 0.0001,
    # batch_size='auto',
    learning_rate = 'adaptive', # 'constant',
    # power_t = 0.5,
     max_iter = 1000, # 200,
    # shuffle = True,
    # random_state = None,
    # tol = 1e-4,
    # verbose = False,
    # warm_start = False,
    # momentum = 0.9,
    # nesterovs_momentum = True,
    # early_stopping = False,
    # validation_fraction = 0.1,
    # beta_1 = 0.9,
    # beta_2 = 0.999,
    # epsilon = 1e-8,
)

Examples = {
   'Bill': {

        'frame': bill,

        'mlpc':mlpc
            },
}