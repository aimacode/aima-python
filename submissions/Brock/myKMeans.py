import csv, numpy, keras_applications

with open('kMeansVoting.csv', newline='') as file:
    data = numpy.array(list(csv.reader(file)))
data[0][0] = '0'
data = data.astype(numpy.float)

Examples = {
    'Voing Data': {
        'data': data,
        'k': [7, 2, 9],
    },
}
