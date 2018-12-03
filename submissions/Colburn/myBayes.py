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
Bball=BayesNet([
    ('StephPlays','',0.96),
    ('KevinPlays','',0.76),
    ('DraymondPlays','',.92),

    ('StephScores+25','StephPlays',{T:.48,F:0.}),
    ('KevinScores+25','KevinPlays',{(T):.56,(F):0.}),
    ('Lose','KevinScores+25 StephScores+25',
     {(T, T): .02,
      (T, F): .06,
      (F, T): .06,
      (F, F): .96}),
    ('Win', 'KevinPlays StephPlays DraymondPlays',
     {(T, T, T): .94,
      (T, T, F): .87,
      (T, F, T): .82,
      (T, F, F): .81,
      (F, T, T): .85,
      (F, T, F): .9,
      (F, F, T): .75,
      (F, F, F): .07
      }),
    ('WinPlayoffGame', 'KevinPlays StephPlays DraymondPlays',
     {(T, T, T): .94,
      (T, T, F): .9,
      (T, F, T): .85,
      (T, F, F): .82,
      (F, T, T): .83,
      (F, T, F): .9,
      (F, F, T): .77,
      (F, F, F): .73})
])

Bball.label='Warriors BasketBall Chances'

'''
{(T, T): .8,
      (T, F): .8,
      (F, T): .917,
      (F, F): .001}),
'''

examples = {


    Bball:[
        {'variable': 'Win',
         'evidence':{'StepPlays':T,'KevinPlays':T,'DraymondPlays':T}},
        {'variable':'Lose',
         'evidence': {'StephScores+25':T,'KevinScores+25':T}},
        {'variable': 'StephPlays',
         'evidence': {'Win':T}},
        # This query seems to give a higher rate than expected, but steph plays in most games so he plays in most loses.
        # Given that we already know its a lose the chance that steph played is still high.
        {'variable': 'StephPlays',
         'evidence': {'Lose':T}},
        {'variable': 'Lose',
         'evidence': {'KevinPlays':T,'DraymondPlays':T}},
        {'variable': 'WinPlayoffGame',
         'evidence': {'KevinPlays':T,'StephPlays':T}},
    ]
}


