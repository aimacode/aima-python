# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

nashvilleWeather = BayesNet([
    ('Rain', '', 0.36),
    ('Freezing', '', 0.27),
    ('CarAccident', 'Rain Freezing',
     {(T, T): 0.95,
      (T, F): 0.79,
      (F, T): 0.29,
      (F, F): 0.001}),
    ('InjuriesReported', 'CarAccident', {T: 0.89, F: 0.09}),
    ('DeathReported', 'CarAccident', {T: 0.95, F: 0.02})
])
nashvilleWeather.label = 'Nashville Weather Correlation With Accidents (Hypothetical)'

examples = {
    nashvilleWeather: [
        {'variable': 'Rain',
         'evidence': {'InjuriesReported': T, 'DeathReported': T},
         },
        {'variable': 'Freezing',
         'evidence': {'InjuriesReported': T, 'DeathReported': T},
         },
        {'variable': 'CarAccident',
         'evidence': {'InjuriesReported': T, 'Rain': F},
         },
        {'variable': 'DeathReported',
         'evidence': {'Rain': T, 'Freezing': T}
        }
    ],
}
