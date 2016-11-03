import traceback

from submissions.Sery import aids


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

aidsECHP = DataFrame()
aidsECHP.data = []
target_data = []

list_of_report = aids.get_reports()
for record in list_of_report:
    try:
        prevalence = float(record['Data']["HIV Prevalence"]["Adults"])
        target_data.append(prevalence)

        year = int(record['Year'])
        living =  int(record['Data']["People Living with HIV"]["Adults"])
        new = int(record['Data']["New HIV Infections"]["Adults"])
        deaths = int(record['Data']["AIDS-Related Deaths"]["Adults"])

        aidsECHP.data.append([year, living, new, deaths])

    except:
        traceback.print_exc()


aidsECHP.feature_names = [
    'Year',
    'People Living with HIV',
    'New HIV Infections',
    'AIDS-Related Deaths',
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
aidsECHP.target = []

def aidsTarget(percentage):
    if percentage > 6:
        return 1
    return 0

for pre in target_data:
    # choose the target
    tt = aidsTarget(pre)
    aidsECHP.target.append(tt)

aidsECHP.target_names = [
    'HIV Prevalence <= 6%',
    'HIV Prevalence > 6%',
]

Examples = {
    'Aids': aidsECHP,
}