import csv

data = []
with open('./forestfires.csv') as csv_data:
    reader = csv.DictReader(csv_data)
    for row in reader:
        this_row = [float(row['temp']), float(row['wind']), float(row['rain']), float(row['RH'])]
        data.append(this_row)

Examples = {
    'ForestFires': {
        'data': data,
        'k': [2, 3, 4]
    }
}

# DataFrame stuff....KMeans didn't like....
# d = {'Temperatures': temps, 'Winds': winds, 'Rains': rain, 'Relative Humidity': humidity}
# df = pd.DataFrame(data=d)
#
# temp_data = df['Temperatures'].values
# wind_data = df['Winds'].values
# rain_data = df['Rains'].values
# humidity_data = df['Relative Humidity'].values
#
# X = np.matrix(zip(temp_data, wind_data, rain_data, humidity_data))
