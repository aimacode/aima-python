import traceback

from submissions.Johnson import education

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

moneyvsthings = DataFrame()

joint = {}

educations = education.get_all_states()
for each in educations:
    try:
        st = each['state']
        expend = each['data']['funding']['expenditures']
        revenue = each['data']['funding']['revenue']
        ratio = each['data']['enrollment']['student teacher ratio']
        eligible = each['data']['enrollment']['students']['other']['free lunch eligible']
        grade8mathscore = each['data']['score']['math'][1]['scale score']
        enrollment = each['data']['enrollment']['students']['all']
        net = revenue - expend
        joint[st] = {}
        joint[st]['ST']= st
        joint[st]['Expend'] = expend
        joint[st]['S/T Ratio'] = ratio
        joint[st]['Net Gain'] = net
        joint[st]['Free Lunch Eligible'] = eligible
        joint[st]['Enrollment'] = enrollment
        joint[st]['8th Grade Math Score'] = grade8mathscore
    except:
        traceback.print_exc()

for st in joint:
    # choose the input values
    moneyvsthings.data.append([
        # countyST,
        # intersection[countyST]['ST'],
        # intersection[countyST]['Trump'],
        #joint[st]['Free Lunch Eligible'],
        joint[st]['S/T Ratio'],
        joint[st]['8th Grade Math Score'],
        joint[st]['Enrollment'],
        #joint[st]['Net Gain']
    ])

moneyvsthings.feature_names = [
    # 'countyST',
    # 'ST',
    # 'Trump',
    #'Free Lunch Eligible',
    'S/T Ratio',
    'Grade 8 Math Scores',
    'Enrollment'
    #'Net Gain'
]


moneyvsthings.target = []

def netpos(number):
    if number > 10000000000:
        return 1
    return 0

for st in joint:
    # choose the target
    ispos = netpos(joint[st]['Expend'])
    moneyvsthings.target.append(ispos)

moneyvsthings.target_names = [
    'Small Expenditure',
    'Large Expenditure',
    #'Free Lunch Eligible <= 300,000',
    #'Free Lunch Eligible > 300,000'
]

Examples = {
    'BestICouldDo': moneyvsthings,
}