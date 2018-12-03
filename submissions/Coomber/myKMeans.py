import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import SpectralClustering


def main():
    with open('autodata.csv', 'r') as f:
      reader = csv.reader(f)
      your_list = list(reader)

    x = np.array(your_list)
    x = x.astype(np.float)


    model = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
                               assign_labels='kmeans')

    labels = model.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=labels,
                s=50, cmap='jet');

    # Normalize // does not effect results
    x = np.divide(x, np.max(x))

    plt.title("KMeans clustering")
    plt.xlabel("miles per gallon")
    plt.ylabel("horse power")
    plt.interactive(False)
    plt.show(block=True)
    def show():
        plt.show()

    return x

x = main()

Examples = {
    'forestFire': {
        'data': x,
        'k': [2, 4, 6],
    },
}
