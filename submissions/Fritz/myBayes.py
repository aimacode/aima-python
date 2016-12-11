import traceback

from submissions.Fritz import medal_of_honor

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

honordata = DataFrame()
honordata.data = []
honortarget = []

class DataFrame2:
    data2 = []
    feature_names2 = []
    target2 = []
    target_names2 = []

honortarget2 = []
honordata2 = DataFrame2()
honordata2.data = []

medalofhonor = medal_of_honor.get_awardees(test=True)

for issued in medalofhonor:
    try:
        date = int(issued['birth']["date"]["year"])
        honortarget.append(date)

        date2 = int(issued['birth']["date"]["month"])
        honortarget2.append(date2)


        day = int(issued['awarded']['date']['day'])
        month = int(issued['awarded']['date']['month'])
        year = int(issued['awarded']['date']['year'])

        #rank = str(issued['military record']['rank'])
        #latitude = str(issued['awarded']["location"]["latitude"])
        #longitude = str(issued['awarded']["location"]["longitude"])

        honordata.data.append([day, month, year])
        honordata2.data.append([day, month, year])

    except:
        traceback.print_exc()

honordata.feature_names = [
    'day',
    'month',
    'year',
]

honordata2.feature_names = [
    'day',
    'month',
    'year',
]


honordata.target = []
honordata2.target = []

def targetdata(HDate):
    if (HDate > 1950 and HDate != -1):
        return 1
    return 0

def targetdata2(HDate2):
    if (HDate2 > 10 and HDate2 != -1):
        return 1
    return 0

for issued in honortarget:

    TD = targetdata(issued)
    honordata.target.append(TD)

honordata.target_names = [
    'Born before 1950',
    'Born after 1950',
]

for issued2 in honortarget2:

    TD2 = targetdata2(issued2)
    honordata2.target.append(TD2)

honordata2.target_names = [
    'Born on or before October',
    'Born after October',
]

Examples = {
    'Year Date': honordata,
    'Month Date': honordata2,
}
