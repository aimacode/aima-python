import csv
import copy as cp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster
import numpy as np
#
# #def main():
# with open('Pokemon.csv','r') as pokemonData:
#     data_iter = csv.reader(pokemonData,
#                            delimiter = ',',
#                            quotechar = '"')
#     data = [data for data in data_iter]
# data_array = np.asarray(data)
#
#
# print(pokemonData)

## composed of generation 1 2 pokemon data, includes all IV, or special attributes every pokemon has

def main():
    Pokedex = []
    with open('Pokemon.csv') as Pokemon:
        PokeDump = csv.reader(Pokemon, delimiter=',')
        count1 = 0
        for row in PokeDump:
            fi = []
            iterate = 0
            if count1 != 0:
                for i in row:
                    if iterate != 0:
                        fi.append(i)
                    iterate += 1
                Pokedex.append(cp.deepcopy(fi))
            count1 += 1
    pca = PCA(n_components=4)
    Normalize = normalize(Pokedex)
    pca.fit(Normalize)
    PokeDexImproved = pca.transform(Normalize)


#visualizing the clusters kmeans. thanks to dummies.com!


    kmeans = cluster.KMeans(n_clusters=4)
    kmeans.fit(PokeDexImproved)
    labels = kmeans.labels_
    types = {0: [], 1: [], 2: [], 3: []}
    for i in range(len(PokeDexImproved)):
        HP = PokeDexImproved[i][1]
        Attack = PokeDexImproved[i][2]
        plot = [HP, Attack]
        types[labels[i]].append(plot)

    for i in range(0, 3):
        types[i] = np.array(types[i])
    plt.xlabel('IV')
    plt.ylabel('Total')

    plt.scatter(types[0][:, 0], types[0][:, 1])
    plt.scatter(types[1][:, 0], types[1][:, 1])
    plt.scatter(types[2][:, 0], types[2][:, 1])
    plt.scatter(types[3][:, 0], types[3][:, 1])
    plt.show()

Start = main()

Examples = {
    'pokemon': {
        'data': Start,
        'k': [2, 5, 6],
    },
}
