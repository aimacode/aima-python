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

College={
    'Differ':''' 
    Ben,Max,Corey,Mary,What,Hawley,Hooper,Who,AI,Physics,Bio,Belmont,Vandy
    ''',
    'kb':'''
Student(Ben,AI,Physics)
Student(Max,AI,Bio)
Student(Corey,ML,Bio)
Student(Mary,Chem,Calc)
Class(AI)
Class(Physics)
Class(Bio)
Professor(Hooper,AI,Belmont)
Professor(Hawley,Physics,Belmont)
Professor(Who,Bio,Belmont)
Professor(What,Calc,Vandy)    
College(Belmont)
College(Vandy)
(Professor(x,c,a) & College(a)) ==> Employed(x,a)
(Professor(x,c,a)) ==> Human(x)
(Student(x,c,b)) ==> Human(x)
(Professor(x,c,y) & Student(a,c,d)) ==> Teaches(x,a)
(Professor(x,c,y) & Student(a,d,c)) ==> Teaches(x,a)
(Teaches(a,b) & Teaches(c,b) & Differ(a,c)) ==> Friends(a,c)
(Teaches(x,a) & Employed(x,y)) ==> Attends(a,y)

''',
    'queries':''' 
Human(x)
Teaches(x,a)
Friends(l,m) 
Employed(x,b)
Attends(x,y)
''',
    'limit': 15
}

Examples = {
    #'farmer': farmer,
    #'weapons': weapons,
    'College': College,
}
