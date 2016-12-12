import traceback

from submissions.Porter import billionaires



class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

Frame = DataFrame()
Frame.data = []




billionaires = billionaires.get_billionaires()


'''
Build the input frame, row by row.
'''
# for countyST in intersection:
#     # choose the input values
#     billionairesPIG.data.append([
#         # countyST,
#         # intersection[countyST]['ST'],
#         # intersection[countyST]['Trump'],
#         intersection[countyST]['Elderly'],
#         intersection[countyST]['College'],
#         intersection[countyST]['Home'],
#         intersection[countyST]['Poverty'],
#     ])

def type(string):
    if string == "new":
        return 1
    if string == "aquired":
        return 2
    else:
        return 0
def political(value):
    if value:
        return 1
    else:
        return 0
def citizen(country):
    if country == 'United States':
        return 1
    else:
        return 0

def gender(string):
    if string == 'male':
        return 1
    else:
        return 0

for x in billionaires:
    Frame.target.append(type(x['company']['type']))

    Frame.data.append([
        political(x['wealth']['how']['was political']),
        citizen(x['location']['citizenship']),
        gender(x['demographics']['gender'])

        ])




Frame.feature_names = ['Political','US Citizen','Male']

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
# billionairesPIG.target = []

# def trumpTarget(percentage):
#     if percentage > 45:
#         return 1
#     return 0

# for countyST in intersection:
#     # choose the target
#     tt = trumpTarget(intersection[countyST]['Trump'])
#     trumpECHP.target.append(tt)

Frame.target_names = ['New', 'Aquired', 'Not Stated']

Examples = {
    'Billionaires': Frame
}