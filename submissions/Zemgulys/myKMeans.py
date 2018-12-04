
data = []

with open('wine_data') as d:
    for line in d:
        inner_list = [elt.strip() for elt in line.split(',')]
        inner_list = [float(i) for i in inner_list]
        data.append(inner_list)

Examples = {
    'wine': {
        'data': data,

        # Maximizes Silhouette Coefficient at a k value of 6
        'k': [5, 6, 7]
    }
}
