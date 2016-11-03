import traceback
from submissions.LaMartina import airlines


class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

flights = DataFrame()

'''
Extract data from the CORGIS airlines.
'''
joint = {}

airline = airlines.get_reports()

for airport in airline:
    try:
        port = airport['airport']['code']
        #weatherdelay is the minutes of weather delay at that airport
        weatherdelay = airport['statistics']['minutes delayed']['weather']
        totalflights = airport['statistics']['flights']['total']
        joint[port] = {}
        joint[port]['Weather Delay Time'] = weatherdelay
        joint[port]['Number of Flights'] = totalflights
    except:
        traceback.print_exc()

flights.data = []

'''
Build the input frame, row by row.
'''
for port in joint:
    # choose the input values
    flights.data.append([
        port,
        joint[port]['Weather Delay Time'],
        joint[port]['Number of Flights'],
    ])


# for county in airlines:
#     try:
#         st = county['Location']['State Abbreviation']
#         countyST = county['Location']['County'] + st
#         trump = county['Vote Data']['Donald Trump']['Percent of Votes']
#         joint[countyST] = {}
#         joint[countyST]['ST']= st
#         joint[countyST]['Trump'] = trump
#     except:
#         traceback.print_exc()
#
# demographics = county_demographics.get_all_counties()
# for county in demographics:
#     try:
#         countyNames = county['County'].split()
#         cName = ' '.join(countyNames[:-1])
#         st = county['State']
#         countyST = cName + st
#         elderly = county['Age']["Percent 65 and Older"]
#         college = county['Education']["Bachelor's Degree or Higher"]
#         home = county['Housing']["Homeownership Rate"]
#         poverty = county['Income']["Persons Below Poverty Level"]
#         if countyST in joint:
#             joint[countyST]['Elderly'] = elderly
#             joint[countyST]['College'] = college
#             joint[countyST]['Home'] = home
#             joint[countyST]['Poverty'] = poverty
#     except:
#         traceback.print_exc()
#
# '''
# Remove the counties that did not appear in both samples.
# '''
# intersection = {}
# for countyST in joint:
#     if 'College' in joint[countyST]:
#         intersection[countyST] = joint[countyST]
#
# trumpECHP.data = []
#
# '''
# Build the input frame, row by row.
# '''
# for countyST in intersection:
#     # choose the input values
#     trumpECHP.data.append([
#         # countyST,
#         # intersection[countyST]['ST'],
#         # intersection[countyST]['Trump'],
#         intersection[countyST]['Elderly'],
#         intersection[countyST]['College'],
#         intersection[countyST]['Home'],
#         intersection[countyST]['Poverty'],
#     ])
#
# trumpECHP.feature_names = [
#     # 'countyST',
#     # 'ST',
#     # 'Trump',
#     'Elderly',
#     'College',
#     'Home',
#     'Poverty',
# ]
#
# '''
# Build the target list,
# one entry for each row in the input frame.
#
# The Naive Bayesian network is a classifier,
# i.e. it sorts data points into bins.
# The best it can do to estimate a continuous variable
# is to break the domain into segments, and predict
# the segment into which the variable's value will fall.
# In this example, I'm breaking Trump's % into two
# arbitrary segments.
# '''
# trumpECHP.target = []
#
# def trumpTarget(percentage):
#     if percentage > 45:
#         return 1
#     return 0
#
# for countyST in intersection:
#     # choose the target
#     tt = trumpTarget(intersection[countyST]['Trump'])
#     trumpECHP.target.append(tt)
#
# trumpECHP.target_names = [
#     'Trump <= 45%',
#     'Trump >  45%',
# ]
# #
# Examples = {
#    # 'Trump': trumpECHP,
#     }