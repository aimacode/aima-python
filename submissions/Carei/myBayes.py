# The probability a patient has cancer
# Bayesian Network Homework
from probability import BayesNet

T, F = True, False

burglary = BayesNet([
    ('Pollution', '', 0.1),
    ('Smoker', '', 0.3),
    ('Cancer', 'Pollution Smoker',
     {(T, T): 0.05,
      (T, F): 0.02,
      (F, T): 0.03,
      (F, F): 0.001}),
    ('XRay', 'Cancer', {T: 0.90, F: 0.20}),
    ('Dyspnoea', 'Cancer', {T: 0.65, F: 0.30})
])
burglary.label = 'Lung Cancer Probability'

examples = {
    burglary: [
        {'variable': 'Cancer',
         'evidence': {'Dyspnoea':T,'Smoker':T},
         },
        {'variable': 'Smoker',
         'evidence': {'Xray':T, 'Cancer':F},
         },
        {'variable': 'Pollution',
         'evidence': {'Dyspnoea':T, 'XRay':T},
         },
        {'variable': 'XRay',
         'evidence': {'Dyspnoea':T, 'Smoker':T},
         },
    ],
}
