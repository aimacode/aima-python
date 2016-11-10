import traceback
from submissions.Ban import county_demographics

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

alumni = DataFrame()
alumni.target = []
alumni.data = []
'''
Extract data from the CORGIS elections, and merge it with the
CORGIS demographics.  Both data sets are organized by county and state.
'''

def alumniTarget(string):
    if (student['Education']["Bachelor's Degree or Higher"] > 50):
        return 1
    return 0

demographics = county_demographics.get_all_counties()
for student in demographics:
    try:
        alumni.target.append(alumniTarget(student['Education']["Bachelor's Degree or Higher"]))

        college = float(student['Education']["Bachelor's Degree or Higher"])
        poverty = float(student['Income']["Persons Below Poverty Level"])

        alumni.data.append([college, poverty])
    except:
        traceback.print_exc()

alumni.feature_names = [
    "Bachelor's Degree or Higher",
    "Persons Below Poverty Level",
]

alumni.target_names = [
    'Most Do Not Have Degree',
    'Most Have Degree',
]

Examples = {
    'Poor with degree': alumni,
}