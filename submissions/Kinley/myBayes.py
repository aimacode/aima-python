import traceback

from submissions.Kinley import drugs


#
#

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []



drugData = DataFrame()
drugData.data = []
targetData = []

alcohol = drugs.get_surveys('Alcohol Dependence')
#tobacco = drugs.get_surveys('Tobacco Use')

i=0

for survey in alcohol[0]['data']:
    try:
        youngUser = float(survey['Young']),
        youngUserFloat = youngUser[0]
        midUser = float(survey['Medium']),
        midUserFloat = midUser[0]
        oldUser = float(survey['Old']),
        oldUserFloat = oldUser[0]
        place = survey['State']

        total = youngUserFloat + midUserFloat + oldUserFloat
        targetData.append(total)

        youngCertain = float(survey['Young CI']),
        youngCertainFloat = youngCertain[0]
        midCertain = float(survey['Medium CI']),
        midCertainFloat = midCertain[0]
        oldCertain = float(survey['Old CI']),
        oldCertainFloat = oldCertain[0]

        drugData.data.append([youngCertainFloat, midCertainFloat, oldCertainFloat])

        i = i + 1
    except:
        traceback.print_exc()


drugData.feature_names = [

      'Young CI',
      'Medium CI',
      'Old CI',
]

drugData.target = []

def drugTarget(number):
    if number > 100.0:
        return 1
    return 0

for pre in targetData:
    # choose the target
    tt = drugTarget(pre)
    drugData.target.append(tt)



drugData.target_names = [

    'States > 100k alcoholics',
    'States < 100k alcoholics',

]

Examples = {
    'Drugs': drugData,
}

# The name of the survey question. Must be one of 'Cocaine Year', 'Alcohol Month',
# 'Cigarette Use', 'Alcohol Risk', 'Illicit/Alcohol Dependence or Abuse', 'Marijuana New',
# 'Illicit Dependence', 'Alcohol Dependence', 'Tobacco Use', 'Alcohol Binge', 'Marijuana Risk',
# 'Alcohol Abuse', 'Marijuana Month', 'Illicit Dependence or Abuse', 'Smoking Risk', 'Illicit Month',
# 'Alcohol Treatment', 'Nonmarijuana Illicit', 'Pain Relievers', 'Marijuana Year',
#  'Illicit Treatment', 'Depression'.

#If you make a typo, it will attempt to suggest a corrected answer. However, this is not perfect, so try to be as accurate as possible.
