from probability import BayesNet

T, F = True, False

compile_error = BayesNet([
    ('MissingSemicolon', '', 0.5),
    ('ExtraSemicolon', '', 0.2),
    ('IndexOutOfRange', '', 0.1),
    ('CompileError', 'MissingSemicolon ExtraSemicolon IndexOutOfRange',
     {
         (T, T, T): 0.2,
         (T, T, F): 0.5,
         (T, F, T): 0.1,
         (T, F, F): 0.7,
         (F, T, T): 0.1,
         (F, T, F): 0.4,
         (F, F, T): 0.75,
         (F, F, F): 0.09
     }),
    ('CantFindSemiColon', 'CompileError', {T: 0.7, F: 0.3}),
    ('ArrayIsNull', 'CompileError', {T: 0.3, F: 0.85}),
    ('OhYeahForgotThat', 'CompileError', {T: 0.8, F: 0.1})
])

compile_error.label = 'Programming Compile Error Correlation With Cause (Hypothetical)'

examples = {
    compile_error: [
        {'variable': 'ExtraSemicolon',
         'evidence': {'CantFindSemiColon': T, 'OhYeahForgotThat': T}
         },
        {'variable': 'MissingSemicolon',
         'evidence': {'ArrayIsNull': F, 'OhYeahForgotThat': T}
         },
        {'variable': 'IndexOutOfRange',
         'evidence': {'ArrayIsNull': T, 'MissingSemicolon': F}
         },
        {'variable': 'ArrayIsNull',
         'evidence': {'IndexOutOfRange': T}
         },
        {'variable': 'OhYeahForgotThat',
         'evidence': {'MissingSemicolon': T, 'ExtraSemicolon': F}
         }
    ]
}