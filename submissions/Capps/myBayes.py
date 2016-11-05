import traceback
from submissions.Capps import graduates

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

graduatesDF= DataFrame()
graduatesDF.target = []
graduatesDF.data = []


'''
Extract data from the CORGIS elections, and merge it with the
CORGIS demographics.  Both data sets are organized by county and state.
'''

def graduateTarget(string):
    if(grads['Employment']["Part Time"] > 200):
        return 1
    return 0

gradData = graduates.get_majors()
for grads in gradData:
    try:
        graduatesDF.target.append(graduateTarget(grads['Employment']["Part Time"]))
        partTime = float(grads['Employment']["Part Time"])
        collegeJobs = float(grads['Earnings']["College Jobs"])

        graduatesDF.data.append([partTime, collegeJobs])
    except:
        traceback.print_exc()


graduatesDF.feature_names = [
    "Part Time",
    "College Jobs",
]

graduatesDF.target_names = [
    'College Jobs get Part Time',
    'College Jobs do not get Part Time',
]

Examples = {
    'GraduatesList': graduatesDF,
}