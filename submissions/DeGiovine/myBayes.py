# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake',
     {(T, T): 0.95,
      (T, F): 0.94,
      (F, T): 0.29,
      (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
])
burglary.label = 'Burglary Example, Fig. 14.2'

examples = {
    burglary: [
        {'variable': 'Burglary',
         'evidence': {'JohnCalls':T, 'MaryCalls':T},
         },
        {'variable': 'Burglary',
         'evidence': {'JohnCalls':F, 'MaryCalls':T},
         },
        {'variable': 'Earthquake',
         'evidence': {'JohnCalls':T, 'MaryCalls':T},
         },
    ],
}
