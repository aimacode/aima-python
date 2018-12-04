
from probability import BayesNet

T, F = True, False

test = BayesNet([
    ('Study', '', 0.3),
    ('Sleep', '', 0.6),
    ('PlayGames', '', 0.1),
    ('Pass', 'Study Sleep PlayGames',
     {(T, T, T): 0.85,
      (T, T, F): 0.98,
      (F, T, T): 0.1,
      (F, F, T): 0.001,
      (F, F, F): 0.01,
      (T, F, T): 0.51,
      (F, T, F): 0.21,
      (T, F, F): 0.78}),
    ('BadDream', 'Pass', {T: 0.4, F: 0.6}),
    ('BeatGame', 'Pass', {T: 0.01, F: 0.2}),
    ('GoodDream', 'Pass', {T: 0.70, F: 0.1})
])
test.label = 'Pass Test'

examples = {
    test: [
        {'variable': 'Study',
         'evidence': {'BadDream':T, 'GoodDream':T},
         },
        {'variable': 'Study',
         'evidence': {'BadDream':F, 'GoodDream':T},
         },
        {'variable': 'PlayGames',
         'evidence': {'BeatGame':F, 'GoodDream':F},
         },
        {'variable': 'PlayGames',
         'evidence': {'BeatGame':T, 'GoodDream':T},
         },
        {'variable': 'Sleep',
         'evidence': {'BadDream':T, 'GoodDream':T},
         },
    ],
}