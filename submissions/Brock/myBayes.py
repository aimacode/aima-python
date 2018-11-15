
from probability import BayesNet

T, F = True, False

depressedStudent = BayesNet([
    ('Sad', '', 0.45),
    ('Homework', '', 0.32),
    ('Tired', '', 0.7),
    ('Sleeping', 'Sad Homework Tired',
     {(T, T, T): 0.12,
      (T, T, F): 0.05,
      (T, F, T): 0.65,
      (T, F, F): 0.33,
      (F, T, T): 0.10,
      (F, T, F): 0.03,
      (F, F, T): 0.89,
      (F, F, F): 0.01}),
    ('Moving', 'Sleeping', {T: 0.5, F: 0.87}),
    ('Drooling', 'Sleeping', {T: 0.56, F: 0.05}),
    ('Outside', 'Sleeping Sad',
     {(T, T): 0.005,
      (T, F): 0.01,
      (F, T): 0.28,
      (F, F): 0.43}),
    ('Exercise', 'Outside Moving',
     {(T, T): 0.6,
      (T, F): 0.0,
      (F, T): 0.34,
      (F, F): 0.0}),
])
depressedStudent.label = 'Depressed Student: is he sleeping, moving, drooling or outside?'

examples = {
    depressedStudent: [
        {'variable': 'Sad',
         'evidence': {'Outside':T, 'Drooling':T},
         },
        {'variable': 'Homework',
         'evidence': {'Outside':F, 'Moving':F},
         },
        {'variable': 'Tired',
         'evidence': {'Outside':F, 'Drooling':T},
         },
        {'variable': 'Sleeping',
         'evidence': {'Outside':T, 'Drooling':F},
         },
        {'variable': 'Exercise',
         'evidence': {'Sad':T, 'Tired':T},
         },
        {'variable': 'Exercise',
         'evidence': {'Sad':F, 'Drooling':T},
         },
    ],
}
