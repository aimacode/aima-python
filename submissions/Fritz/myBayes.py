import traceback

from submissions.Fritz import medal_of_honor

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

dataframe = DataFrame()

medalofhonor = medal_of_honor.get_awardees(test=True)

for issued in medalofhonor:
    try:
        #awardee = name['name']
        date = 'issued'

    except:
        traceback.print_exc()

'''
Build the input frame, row by row.
'''
for issued in medalofhonor:
    # choose the input values
    dataframe.data.append([
        # date awarded,
        date,
    ])

dataframe.feature_names = [
    # 'date awarded-',
    'Date Awarded',
]

dataframe.target = []

def targetDate(totalDate):
    if totalDate.__contains__('1990'):
        return 1
    return 0

for issued in medalofhonor:
    # choose the target
    TD = targetDate(date)
    dataframe.target.append(TD)

dataframe.target_names = [
    'Date is in 1990',
    'Date is not in 1990',
]

Examples = {
    'Date': dataframe,
}