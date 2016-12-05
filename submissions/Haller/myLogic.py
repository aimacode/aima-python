school = {
    'kb': '''
Teacher(Hooper)
Teacher(Crowell)
Department(Miller)
Teacher(Nobody)
Student(Patrick)
Student(Austin)
Dean(Spence)
College(Spence, CSM)
Boss(Miller, Hooper)
Boss(Miller, Crowell)
Boss(Spence, Miller)
Class(Hooper, Patrick)
Class(Hooper, Austin)
Class(Hooper, Anderson)
Class(Hooper, Capps)
Class(Hooper, Johnson)
Class(Garland, Patrick)
Class(Crowell, Johnson)
Class(Miller, Patrick)


(Class(t, s)) ==> Student(s)
(Class(t, s)) ==> Teacher(t)
(College(p, c) & Boss(p, l)) ==> College(l, c)
(Boss(b, s) & Boss(s, t)) ==> BigCheese(b, t)

''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
BigCheese(x, y)
Class(Hooper, x)
Class(x, Patrick)
Teacher(x)
College(x, CSM)
Student(x)
''',
    'limit': 4,
}

Examples = {
    'school': school,
}