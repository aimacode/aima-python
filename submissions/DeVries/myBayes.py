from probability import BayesNet

T, F = True, False

# cards = BayesNet([
#     ('KingOfHearts', '', 0.01923),
#     ('RedCard', '', 0.5),
#     ('RedKing', 'KingOfHearts RedCard',
#      {(T, T): 0.0192,
#       (T, F): 0.02,
#       (F, T): 0.0368,
#       (F, F): 0.04}),
#     ('RedOnTop', 'RedKing', {T: 0.03846, F: 0.076923}),
#     # ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
# ])
# cards.label = 'Cards'

smoking = BayesNet([
    ('Female', '', 0.51),
    ('LGB', '', 0.054),
    ('Smoker', 'LGB Female',
     {(T, T): 0.3047,
      (T, F): 0.3046,
      (F, T): 0.1885,
      (F, F): 0.2202}),
    ('LungCancer', 'Smoker Female',
     {(T, T): 0.095,
      (T, F): 0.159,
      (F, T): 0.004,
      (F, F): 0.002}),
    ('PrematureDeath', 'LungCancer', {T: 0.86, F: 0.0134}),
    # ('Smoker', 'LGB', {T: 0.205, F: 0.153})
])
smoking.label = 'Smoking Example'

examples = {
    smoking: [
        {'variable': 'LungCancer',
         'evidence': {'LGB':T},
         },
        {'variable': 'LGB',
         'evidence': {'PrematureDeath':F},
         },
        {'variable': 'PrematureDeath',
         'evidence': {'Female':T, 'LGB':T},
         },
        {'variable': 'Smoker',
         'evidence': {'Female':F, 'PrematureDeath':T},
         },
    ],
}
