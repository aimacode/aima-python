import traceback
from submissions.VanderKallen import slavery

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

slaveTot = DataFrame()
slaveAS = DataFrame
transactions = slavery.get_transaction()
slaveTot.data = []
slaveAS.data = []

def genderNumber(gender):
    if gender == 'M':
        return 1
    else:
        return 0

for seller in transactions:
    # choose the input values
    slaveTot.data.append([
        seller['Transaction']['Number of Child Slaves'],
        seller['Transaction']['Number of Adult Slaves'],
        ])
    slaveAS.data.append([
        seller['Slave']['Age'],
        genderNumber(seller['Slave']['Gender']),
    ])

slaveAS.feature_names = [
    'Age',
    'Gender',
]

slaveTot.feature_names = [
    'Children',
    'Adults',

]

slaveTot.target = []
slaveAS.target = []

def priceTarget(price):
    if price < 410:
        return 1
    return 0

for deal in transactions:
    # choose the target
    tt = priceTarget(deal['Transaction']['Sale Details']['Price'])
    slaveTot.target.append(tt)
    slaveAS.target.append(tt)

slaveTot.target_names = [
    'Price <= $410',
    'Price >  $410',
]

slaveAS.target_names = [
    'Price <= $410',
    'Price >  $410',
]

Examples = {
    'Sales by Children and Adults': slaveTot,
    'Sales by Age and Sex': slaveAS
}