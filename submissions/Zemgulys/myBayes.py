# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

coffee = BayesNet([
    ('Tired', '', 0.24),
    ('Working', '', 0.45),
    ('Coffee', 'Tired Working',
     {(T, T): 0.97,
      (T, F): 0.38,
      (F, T): 0.29,
      (F, F): 0.05}),
    ('Energy', 'Coffee', {T: 0.83, F: 0.21}),
    ('Jitters', 'Coffee', {T: 0.19, F: 0.01})
])
coffee.label = 'Probability of getting coffee'

examples = {
    coffee: [
        {'variable': 'Tired',
         'evidence': {'Energy':T, 'Jitters':T},
         },
        {'variable': 'Tired',
         'evidence': {'Energy':F, 'Jitters':T},
         },
        {'variable': 'Working',
         'evidence': {'Energy':T, 'Jitters':T},
         },
    ],
}
