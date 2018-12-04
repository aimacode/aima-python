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

storm = BayesNet([
    ('Thunderstorm', '', 0.005),
    ('Earthquake', '', 0.002),
    ('Rain', '', 0.02),
    ('Loud', 'Thunderstorm Earthquake Rain',
     {(T, T, T): 0.99,
      (T, F, T): 0.65,
      (T, T, F): 0.95,
      (T, F, F): 0.85,
      (F, T, T): 0.80,
      (F, T, F): 0.50,
      (F, F, T): 0.20,
      (F, F, F): 0.05,
      }),
    ('Dog Barks', 'Loud', {T: 0.90, F: 0.10}),
    ('Cat Hides', 'Loud', {T: 0.80, F: 0.40}),
])

storm.label = 'What is causing the noise?'

examples = {
    storm: [
        {'variable': 'Thunderstorm',
         'evidence': {'Dog Barks':T, 'Cat Hides':T},
         },
        {'variable': 'Earthquake',
         'evidence': {'Dog Barks':F, 'Cat Hides':T},
         },
        {'variable': 'Rain',
         'evidence': {'Dog Barks':T, 'Cat Hides':T},
         },
        {'variable': 'Thunderstorm',
         'evidence': {'Dog Barks':F, 'Cat Hides':F},
         },
        {'variable': 'Earthquake',
         'evidence': {'Dog Barks':T, 'Cat Hides':F},
         },
        {'variable': 'Rain',
         'evidence': {'Dog Barks':F, 'Cat Hides':F},
         },
    ],

}