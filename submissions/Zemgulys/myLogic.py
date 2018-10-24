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

Bands = {
    'kb': '''
Band(LedZeppelin)
Band(TheWho)

Guitarist(Jimmy, LedZeppelin) 
Drummer(JB, LedZeppelin)
Bassist(JPJ, LedZeppelin)
Vocalist(Robert, LedZeppelin)

Guitarist(Pete, TheWho) 
Drummer(Keith, TheWho)
Bassist(JE, TheWho)
Vocalist(Roger, TheWho)

Guitarist(x, y) ==> Member(x, y)
Drummer(x, y) ==> Member(x, y)
Bassist(x, y) ==> Member(x, y)
Vocalist(x, y) ==> Member(x, y)

Member(x, TheWho) & Member(y, LedZeppelin) ==> Hates(x, y)
Member(x, LedZeppelin) & Member(y, TheWho) ==> Hates(x, y)
 
''',

    'queries': '''
Member(x, LedZeppelin)
Member(x, TheWho)
Hates(x, Jimmy)
Hates(x, Roger)
''',
    'limit': 20
}

Examples = {
    # 'farmer': farmer,
    # 'weapons': weapons,
    'Bands': Bands
}