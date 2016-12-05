import traceback
from submissions.Hess import cars

class DataFrame:
    data = []
    feature_names = []
    target = []
    target_names = []

guzzle = DataFrame()
guzzle.target = []
guzzle.data = []

guzzle = cars.get_cars()

def guzzleTarget(string):
    if (info['Fuel Information']['City mph'] < 14):
        return 1
    return 0

for info in guzzle:
    try:

        guzzle.data.append(guzzleTarget(info['Fuel Information']['City mph']))
        fuelCity = float(info['Fuel Information']['City mph']) # they misspelled mpg
        year = float(info['Identification']['Year'])

        guzzle.data.apend([fuelCity, year])
    except:

        traceback.print_exc()

guzzle.feature_names = [
    "City mph"
    "Year"
]

guzzle.target_names = [
    "New Car is < 14 MPG"
    "New Car is > 14 MPG"
]

Examples = {
    'New car that guzzles': guzzle
}
