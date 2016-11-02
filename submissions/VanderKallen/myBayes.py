import traceback
from submissions.VanderKallen import slavery

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

slavevvv = DataFrame()

transactions = slavery.get_transaction()
slavevvv.data = []

def genderNumber(gender):
    if gender == 'M':
        return 1
    else:
        return 0

for seller in transactions:
    # choose the input values
    slavevvv.data.append([
        # countyST,
        # intersection[countyST]['ST'],
        # intersection[countyST]['Trump'],
        #seller['Transaction']['Number of Child Slaves'],
        #seller['Transaction']['Number of Adult Slaves'],
        seller['Transaction']['Number of Total Slaves Purchased'],
        seller['Slave']['Age'],
        #genderNumber(seller['Slave']['Gender']),
    ])

slavevvv.feature_names = [
    # 'countyST',
    # 'ST',
    # 'Trump',
    #'Elderly',
    #'College',
    #'Home',
    #'Children',
    #'Adults',
    'Total',
    'Age',
    #'Gender',
]

'''
Build the target list,
one entry for each row in the input frame.

The Naive Bayesian network is a classifier,
i.e. it sorts data points into bins.
The best it can do to estimate a continuous variable
is to break the domain into segments, and predict
the segment into which the variable's value will fall.
In this example, I'm breaking Trump's % into two
arbitrary segments.
'''
slavevvv.target = []

def priceTarget(price):
    if price > 200:
        return 1
    return 0

for deal in transactions:
    # choose the target
    tt = priceTarget(deal['Transaction']['Sale Details']['Price'])
    slavevvv.target.append(tt)

slavevvv.target_names = [
    'Price <= 1000',
    'Price >  1000',
]

Examples = {
    'Slavery': slavevvv,
}