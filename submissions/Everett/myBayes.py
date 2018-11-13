# Cancer example
from probability import BayesNet

T, F = True, False

cancer = BayesNet([
    ('Cancer', '', 0.001),
    ('Hereditary', '', 0.002),
    ('GeneMutations', 'Cancer Hereditary',
     {(T, T): 0.95,
      (T, F): 0.94,
      (F, T): 0.29,
      (F, F): 0.001}),
    ('Smoking', 'GeneMutations', {T: 0.90, F: 0.05}),
    ('Carcinogens', 'GeneMutations', {T: 0.70, F: 0.01})
])
cancer.label = 'Cancer Example'

examples = {
    cancer: [
        {'variable': 'Cancer',
         'evidence': {'Smoking':T, 'Carcinogens':T},
         },
        {'variable': 'Cancer',
         'evidence': {'Smoking':F, 'Carcinogens':T},
         },
        {'variable': 'Hereditary',
         'evidence': {'Smoking':T, 'Carcinogens':T},
         },
    ],
}
