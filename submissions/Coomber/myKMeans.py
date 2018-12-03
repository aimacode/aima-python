import csv
import numpy as np

with open('forestfiresdata.csv', 'r') as f:
  reader = csv.reader(f)
  your_list = list(reader)

  x = np.array(your_list)
  y = x.astype(np.float)

Examples = {
    'forestFire': {
        'data': y,
        'k': [2, 3, 4],
    },
}
