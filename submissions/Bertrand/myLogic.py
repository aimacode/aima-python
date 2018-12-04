farmer = {
    'kb': '''
Farmer(Mac)
Rabbit(Pete)
Mother(MrsMac, Mac)
Mother(MrsRabbit, Pete)
(Rabbit(r) & Farmer(f)) ==> Hates(f, r)
(Mother(m, c)) ==> Loves(m, c)
(Mother(m, r) & Rabbit(r)) ==> Rabbit(m)
(Farmer(f)) ==> Human(f)
(Mother(m, h) & Human(h)) ==> Human(m)
''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Human(x)
Hates(x, y)
''',
#    'limit': 1,
}

weapons = {
    'kb': '''
(American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)
Owns(Nono, M1)
Missile(M1)
(Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)
Missile(x) ==> Weapon(x)
Enemy(x, America) ==> Hostile(x)
American(West)
Enemy(Nono, America)
''',
    'queries':'''
Criminal(x)
''',
}

romeo_and_juliet = {
    'kb': '''
(Capulet(x) & Montague(y)) ==> Hates(x, y)
Montague(LadyMontague)
Child(Romeo, LadyMontague)
(Child(m, f) & Montague(f)) ==> Montague(m)
Capulet(Juliet)
Parent(SirCapulet, Juliet)
(Parent(s, j) & Capulet(j)) ==> Capulet(s)
(Married(p, q) & Capulet(q)) ==> Capulet(p)
Married(LadyCapulet, SirCapulet)
(Capulet(n) & Capulet(m)) ==> Loves(n, m)
(Montague(n) & Montague(m)) ==> Loves(n, m) 
''',
    'queries':'''
Montague(x)
Capulet(s)
Hates(x,y)
Loves(n,m)
''',
    'limit': 20,
}

Examples = {
    'romeo_and_juliet': romeo_and_juliet,
}