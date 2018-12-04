
from probability import BayesNet

T, F = True, False

grass = BayesNet([
    ('Raining', '', 0.1),
    ('Sprinklers', '', 0.4),
    ('GrassWet', 'Raining Sprinklers',
     {(T, T): 0.99,
      (T, F): 0.82,
      (F, T): 0.94,
      (F, F): 0.001}),
    ('DadIsHappy', 'GrassWet', {T: 0.90, F: 0.05}),
    ('AquaphobicNeighborIsNotHappy', 'GrassWet', {T: 0.70, F: 0.01})
])
grass.label = 'WetGrass'

examples = {
    grass: [
        {'variable': 'Raining',
         'evidence': {'DadIsHappy': T, 'AquaphobicNeighborIsNotHappy': T},
         },
        {'variable': 'Sprinklers',
         'evidence': {'DadIsHappy': F, 'AquaphobicNeighborIsNotHappy': T},
         },
        {'variable': 'Raining',
         'evidence': {'DadIsHappy': T, 'AquaphobicNeighborIsNotHappy': F},
         },
    ],
}
