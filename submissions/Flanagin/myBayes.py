from probability import BayesNet

T, F = True, False

cat = BayesNet([
    # things I am doing
    ('HW', '', .65),
    ('Bed', '', 0.4),
    ('Laptop', '', 0.3),
    # if she sits on me based on what I'm doing
    ('CatSits', 'HW Bed Laptop',
     {(T, T, T): 0.35,
      (T, T, F): 0.8,
      (T, F, T): 0.3,
      (T, F, F): 0.85,
      (F, T, T): 0.62,
      (F, T, F): 0.9,
      (F, F, T): 0.7,
      (F, F, F): 0.2}),
    # my reaction to my cat sitting on me
    ('Pets', 'CatSits', {T: 0.7, F: 0.03}),
    ('Feeds', 'CatSits', {T: .2, F: 0.85})
])
cat.label = "Interactions with my cat"

examples = {
    cat: [
        {'variable': 'HW',
         'evidence': {'Pets': T, 'Feeds': F}
         },
        {'variable': 'HW',
         'evidence': {'Pets': F, 'Feeds': F}
         },
        {'variable': 'Bed',
         'evidence': {'Pets': T, 'Feeds': T}
         },
        {'variable': 'Laptop',
         'evidence': {'Pets': F, 'Feeds': T}
         }

    ]
}