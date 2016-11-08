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

classroom = {
    'kb':'''
    Teacher(Sally)
    Education(Sally,PHD)
    Education(Bob,Bachelors)
    Action(Lisa,yells)
    Student(Lisa)
    Student(John)
    Prepared(John)
    OnTime(John)
    TA(Bob)



    (Student(x) & Action(x,yells)) ==> Argues(x,Sally)
    (Argues(x,y)) ==> Detention(y,x)
    (Student(x) & GoodStudent(x)) ==> Pass(x)
    (Student(x) & Prepared(x) & OnTime(x))==> GoodStudent(x)
    (Teacher(x)) ==> Human(x)
    (Student(x)) ==> Human(x)
    (TA(x)) ==> Human(x)
    (Education(x,PHD) & Human(x)) ==> Teacher(x)
    (Education(x,Bachelors) & Human(x)) ==> TA(x)






    ''',
    'queries':'''
    Teacher(x)
    Detention(y,x)
    Pass(x)
    TA(x)

    ''',
}



Examples = {
    # 'farmer': farmer,
    # 'weapons': weapons,
    'classroom': classroom,
}