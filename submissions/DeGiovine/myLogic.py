
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


vehicles = {
    'kb': '''
Company(Lexus, Japan)
Company(Chevy, America)
Company(Ducati, Italy)
Company(Lamborghini, Italy)
Company(Volkswagen, Germany)

Parent(Toyota, Lexus)
Parent(GM, Chevy)
Parent(Lamborghini, Ducati)
Parent(Volkswagen, Lamborghini)

Product(Car, Lexus)
Product(Car, Chevy)
Product(Bike, Ducati)
Product(Car, Lamborghini)
Product(Car, Volkswagen)

(Company(c, h) & Product(p,c)) ==> Sells(c, p)
(Parent(p, c) & Company(c, h)) ==> ParentCompanies(p)
(Company(c, h)) ==> Headquarters(c, h)
(Company(c, h)) ==> Companies(c)
(Car(a)) ==> Vehicle(v)   
(Bike(b)) ==> Vehicle(v) 

''',

    'queries': '''
Companies(x)
ParentCompanies(x)
Parent(p, c)
Sells(c, p)
Headquarters(c, h)
''',
}


Examples = {
    'farmer': farmer,
    'weapons': weapons,
    'vehicles': vehicles,
}







# weights2d = np.array([
#    [1, 0, 0],
#    [0, 1, 0],
#    [0, 0, 1],
