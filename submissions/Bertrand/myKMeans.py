import csv
import numpy as np

# import data from csv file
with open('airlines.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = np.asarray(data)

# remove columns containing non numerical data
data = np.delete(data, np.s_[7,11,19,20], axis=1)
data = np.delete(data, np.s_[0], axis=0)    # remove column headers
data = data.astype(float)   # convert strings to floats
Examples = {
    'Airline Data': {
        'data': data,
        'k': [2, 3, 4],
    },
}
