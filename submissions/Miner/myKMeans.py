import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import math

columns = ['Temperature', 'Wind', 'Rain', 'Humidity']
data = []
with open('./forestfires.csv') as csv_data:
    reader = csv.DictReader(csv_data)
    for row in reader:
        this_row = [float(row['temp']), float(row['wind']), float(row['rain']), float(row['RH'])]
        data.append(this_row)

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

colormap = np.array(['red', 'lime', 'black', 'blue'])
centers = kmeans.cluster_centers_
plt.figure(figsize=(14, 7))

# temps = []
# winds = []
# for entry in data:
#     temps.append(entry[0])
#     winds.append(entry[1])

# Plot the Models Classifications
plt.subplot(1, 2, 1)
# plt.scatter(data[0], data[1], c=colormap[kmeans.labels_], s=40)
plt.title('K Mean Classification')

for x in data:
    distances = []
    # Temperature and Wind
    for num in range(2):
        d = math.sqrt((x[0] - centers[num][0])**2
                      + (x[1] - centers[num][1])**2)
        distances.append(d)
    close = distances.index(min(distances))
    color = colormap[close]
    plt.plot(x[0], x[1], 'o', color=color)


def show():
    plt.show()


Examples = {
    'ForestFires': {
        'data': data,
        'k': [2, 3, 4],
        'main': show
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
