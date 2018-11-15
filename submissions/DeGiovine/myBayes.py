# Burglary example [Figure 14.2]
from probability import BayesNet

T, F = True, False

CarAccidents = BayesNet([
    ('Texting', '', 0.002),
    ('Speeding', '', 0.002),
    ('Weather', '', 0.001),
    ('Alcohol', '', 0.003),
    ('AccidentA', 'Texting Speeding Weather Alcohol',
     {(T, T, T, T): 0.90,
      (F, T, T, T): 0.74,
      (T, F, T, T): 0.59,
      (T, T, F, T): 0.79,
      (T, T, T, F): 0.58,
      (F, F, T, T): 0.43,
      (F, T, F, T): 0.63,
      (F, T, T, F): 0.48,
      (T, F, F, T): 0.42,
      (T, F, T, F): 0.27,
      (T, T, F, F): 0.47,
      (T, F, F, F): 0.16,
      (F, T, F, F): 0.31,
      (F, F, T, F): 0.11,
      (F, F, F, T): 0.32,
      (F, F, F, F): 0.005}),

    # Attempt to try separating each kind of accident
    # ('AccidentB', 'Texting Weather',
    #  {(T, T): 0.37,
    #   (T, F): 0.16,
    #   (F, T): 0.11,
    #   (F, F): 0.05}),
    # ('AccidentC', 'Texting Alcohol',
    #  {(T, T): 0.58,
    #   (T, F): 0.16,
    #   (F, T): 0.32,
    #   (F, F): 0.05}),
    # ('AccidentD', 'Speeding Weather',
    #  {(T, T): 0.52,
    #   (T, F): 0.31,
    #   (F, T): 0.11,
    #   (F, F): 0.05}),
    # ('AccidentE', 'Speeding Alcohol',
    #  {(T, T): 0.73,
    #   (T, F): 0.31,
    #   (F, T): 0.32,
    #   (F, F): 0.05}),
    # ('AccidentF', 'Weather Alcohol',
    #  {(T, T): 0.53,
    #   (T, F): 0.11,
    #   (F, T): 0.32,
    #   (F, F): 0.05}),

    ('HitCar', 'AccidentA', {T: 0.40, F: 0.04}),
    ('HitGuardRail', 'AccidentA', {T: 0.50, F: 0.02}),

    # Attempt for each of the aforementioned to have their own probabilities
    # ('HitCar', 'AccidentB', {T: 0.60, F: 0.40}),
    # ('HitGuardRail', 'AccidentB', {T: 0.80, F: 0.20}),
    #
    # # ('HitCar', 'AccidentC', {T: 0.60, F: 0.40}),
    # ('HitGuardRail', 'AccidentC', {T: 0.80, F: 0.20}),
    #
    # ('HitCar', 'AccidentD', {T: 0.60, F: 0.40}),
    # ('HitGuardRail', 'AccidentD', {T: 0.80, F: 0.20}),
    #
    # ('HitCar', 'AccidentE', {T: 0.60, F: 0.40}),
    # ('HitGuardRail', 'AccidentE', {T: 0.80, F: 0.20}),
    #
    # ('HitCar', 'AccidentF', {T: 0.60, F: 0.40}),
    # ('HitGuardRail', 'AccidentF', {T: 0.80, F: 0.20})
])
CarAccidents.label = 'Car Accident Example, Fig. 1'

examples = {
    CarAccidents: [
        # Texting
        {'variable': 'Texting',
         'evidence': {'HitCar':T, 'HitGuardRail':T},
         },
        {'variable': 'Texting',
         'evidence': {'HitCar':F, 'HitGuardRail':T},
         },
        {'variable': 'Texting',
         'evidence': {'HitCar':T, 'HitGuardRail':F},
         },

        # Speeding
        {'variable': 'Speeding',
         'evidence': {'HitCar':T, 'HitGuardRail':T},
         },
        {'variable': 'Speeding',
         'evidence': {'HitCar':F, 'HitGuardRail':T},
         },
        {'variable': 'Speeding',
         'evidence': {'HitCar':T, 'HitGuardRail':F},
         },

        # Weather
        {'variable': 'Weather',
         'evidence': {'HitCar':T, 'HitGuardRail':T},
         },
        {'variable': 'Weather',
         'evidence': {'HitCar':F, 'HitGuardRail':T},
         },
        {'variable': 'Weather',
         'evidence': {'HitCar':T, 'HitGuardRail':F},
         },

        # Alcohol
        {'variable': 'Alcohol',
         'evidence': {'HitCar':T, 'HitGuardRail':T},
         },
        {'variable': 'Alcohol',
         'evidence': {'HitCar':F, 'HitGuardRail':T},
         },
        {'variable': 'Alcohol',
         'evidence': {'HitCar':T, 'HitGuardRail':F},
         },
    ],
}
