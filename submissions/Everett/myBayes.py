# Cancer example
from probability import BayesNet

T, F = True, False

cancer = BayesNet([
    ('Pollution', '', 0.10),
    ('Smoker','', 0.90),
    ('LungCancer', 'Pollution Smoker',
     {(T, T): 0.987,
      (T, F): 0.10,
      (F, T): 0.87,
      (F, F): 0.09}),
    ('XRay', 'LungCancer', {T: 0.90, F: 0.10}),
    ('Dyspnoea', 'LungCancer', {T: 0.70, F: 0.30}),
    ('Death', 'LungCancer', {T: 0.87, F: 0.13}),
])
cancer.label = 'Lung Cancer Example'

examples = {
    cancer: [
        {'variable': 'LungCancer',
         'evidence': {'Death':T, 'Pollution':T},
         },
        {'variable': 'Death',
         'evidence': {'LungCancer':F, 'Smoker':T},
         },
        {'variable': 'Smoker',
         'evidence': {'LungCancer':T, 'Pollution':T},
         },
        {'variable': 'Pollution',
         'evidence': {'LungCancer':T, 'Xray':T},
        },
         {'variable': 'Smoker',
         'evidence': {'LungCancer':F, 'Dyspnoea':T, 'XRay':T},
        }

    ],

}
