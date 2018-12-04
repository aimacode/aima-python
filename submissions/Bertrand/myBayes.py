# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

death = BayesNet([
    ('Overweight', '', 0.339),
    ('HighStress', '', 0.72),
    ('HeartAttack', 'Overweight HighStress',
     {(T, T): 0.72,
      (T, F): 0.15,
      (F, T): 0.12,
      (F, F): 0.003}),
    ('OldAge', '', 0.147),
    ('Death', 'HeartAttack OldAge',
     {(T, T): 0.749,
      (T, F): 0.25,
      (F, T): 0.0009,
      (F, F): 0.001})
])
death.label = 'Death and Heart Attacks'

examples = {
    death: [
        {'variable': 'Overweight',
         'evidence': {'Death':T,},
         },
        {'variable': 'Death',
         'evidence': {'HeartAttack':T, 'OldAge':F},
         },
        {'variable': 'HighStress',
         'evidence': {'Death':F, },
         },
        {'variable': 'OldAge',
         'evidence': {'HeartAttack':T, },
        },
        {'variable': 'OldAge',
         'evidence': {'HighStress': F, 'Overweight': T},
        },
    ],
}
