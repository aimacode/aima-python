from probability import BayesNet

T, F = True, False

snowDay = BayesNet([
    ('snow', '', 0.022),
    ('windChill', '', 0.017),
    ('advisory', 'snow windChill',
     {(T, T): 0.62,
      (T, F): 0.48,
      (F, T): 0.41,
      (F, F): 0.0001}),
    ('noSchool', 'advisory', {T: 0.70, F: 0.005}),
    ('extremeWarning', 'advisory', {T: 0.16, F: 0.002})
])
snowDay.label = 'snowDay shows various weather probabilities that effect the chances of a snow day in Buffalo, New York.'

examples = {
    snowDay: [
        {'variable': 'snow',
         'evidence': {'noSchool':T, 'extremeWarning':T},
         },
        {'variable': 'windChill',
         'evidence': {'noSchool': F, 'advisory': T},
         },
        {'variable': 'advisory',
         'evidence': {'noSchool': T, 'windChill': F},
         },
        {'variable': 'extremeWarning',
         'evidence': {'snow': T, 'windChill': T},
         },

    ],
}
