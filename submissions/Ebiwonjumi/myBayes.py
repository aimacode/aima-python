# Accidents on the Nigerian Road
from probability import BayesNet

T, F = True, False

Aids = BayesNet([
    ('HIV', '', 0.1),
    ('Aids', '', 0.000982),
    ('Over30', '', 0.9),
    ('Dead', 'HIV Aids Over30',
     {
         (T, T, T): 0.90,
         (T, T, F): 0.13,
         (T, F, T): 0.85,
         (T, F, F): 0.07,
         (F, T, T): 0.0,
         (F, T, F): 0.0,
         (F, F, T): 0.99,
         (F, F, F): 0.95


     }),
    ('Jason', 'Dead', {T: 0.01, F: 0.90}),
    ('Lauren', 'Dead', {T: 0.20, F: 0.50}),
    ('Zach', 'Dead', {T: 0.49, F: 0.11})
])
Aids.label = 'Probability of Death with HIV/AIDS'

examples = {
    Aids: [
        {'variable': 'HIV',
         'evidence': {'Jason':T, 'Lauren':T, 'Zach':T},
         },
        {'variable': 'HIV',
         'evidence': {'Jason':F, 'Lauren':T, 'Zach':T},
         },
        {'variable': 'HIV',
         'evidence': {'Jason':F, 'Lauren':F, 'Zach':T},
         },
        {'variable': 'HIV',
         'evidence': {'Jason':F, 'Lauren':F, 'Zach':F},
         },
        {'variable': 'HIV',
         'evidence': {'Jason':F, 'Lauren':T, 'Zach':F},
         },
        {'variable': 'Aids',
         'evidence': {'Jason':T, 'Lauren':T, 'Zach':T},
         },
        {'variable': 'Aids',
         'evidence': {'Jason':F, 'Lauren':T, 'Zach':T},
         },
        {'variable': 'Aids',
         'evidence': {'Jason':F, 'Lauren':F, 'Zach':T},
         },
        {'variable': 'Aids',
         'evidence': {'Jason':F, 'Lauren':F, 'Zach':F},
         },
        {'variable': 'Aids',
         'evidence': {'Jason':F, 'Lauren':T, 'Zach':F},
         },
    ],
}
