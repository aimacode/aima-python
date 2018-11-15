# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

gym = BayesNet([
    #WorkOuts
    ('LegDay', '', .33),
    ('ArmsDay', '', 0.33),
    ('Cardio', '', 0.33),
    ('Tired', 'LegDay ArmDay', 'Cardio',
     {(T, T, T): 0.1,
      (T, T, F): 0.1,
      (T, F, T): 0.7,
      (T, F, F): 0.8,
      (F, T, T): 0.7,
      (F, T, F): 0.9,
      (F, F, T): 0.9,
      (F, F, F): 0.5}),
    ('Quit', 'Tired', {T: 0.70, F: 0.01}),
    ('Push', 'Tired', {T: 0.90, F: 0.10})
])
gym.label = 'Gym Day'

examples = {
    gym: [
        {'variable': 'Legday',
         'evidence': {'Quit': T, 'Push': F}
         },
        {'variable': 'Legday',
         'evidence': {'Quit': F, 'Push': F}
         },
        {'variable': 'Armday',
         'evidence': {'Quit': T, 'Push': T}
         },
        {'variable': 'Cardio',
         'evidence': {'Quit': F, 'Push': T}
         }
    ]
}
