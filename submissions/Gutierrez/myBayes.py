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

famous = BayesNet([
# will you get famous?
# statistics recevied from https://nypost.com/2009/07/25/you-are-not-going-to-be-famous/ and
    ('liveInAmerica', '', 0.04),
    ('liveInNashville', '', 0.002),
    ('Famous', 'liveInAmerica liveInNashville',
     {(T, T): 0.000000024,
      (T, F): 0.000000024,
      (F, T): 0.0000000012,
      (F, F): 0.0000006}),
    ('Actor', 'Famous', {T: 0.00041, F:0.9}),
    ('Musician', 'Famous', {T: 0.0002, F: 0.9})
])
famous.label = 'will you get famous'


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
    famous: [
        {'variable': 'liveInNashville',
         'evidence': {'Actor':T, 'Musician':T},
         },
        {'variable': 'livesInNashville',
         'evidence': {'Actor':F, 'Musician':T},
         },
        {'variable': 'liveInAmerica',
         'evidence': {'Actor':T, 'Musician':T},
         },
    ],


}
