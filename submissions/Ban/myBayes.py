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
        alumni.target.append(alumniTarget(student['Education']["High School or Higher"]))

        college = float(student['Education']["High School or Higher"])
        poverty = float(student['Income']["Persons Below Poverty Level"])
        ethnicity = float(student['Ethnicities']["White Alone"])

        alumni.data.append([college, poverty, ethnicity])
    except:
        traceback.print_exc()

alumni.feature_names = [
    "High School or Higher",
    "Persons Below Poverty Level",
    "White Alone",
]

alumni.target_names = [
    'Most did not graduate college',
    'Most did graduate college',
    'Ethnicity',
]

Examples = {
    'Poor with degree': alumni,
}