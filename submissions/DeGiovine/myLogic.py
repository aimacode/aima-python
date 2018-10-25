
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

Industry(Car, Lexus)
Industry(Car, Chevy)
Industry(Bike, Ducati)
Industry(Car, Lamborghini)
Industry(Car, Volkswagen)

(Company(c, h) & Product(p,c)) ==> Sells(c, p)
(Parent(p, c) & Company(c, h)) ==> ParentCompanies(p)
(Company(c, h)) ==> CountriesOfOrigin(h)
(Company(c, h)) ==> Make(c)
(Company(c, h)) ==> Headquarters(c, h)
(Car(a)) ==> Vehicle(v)   
(Car(a)) ==> Cars(s) 
(Bike(b)) ==> Vehicle(v) 
(Bike(b)) ==> Bikes(m)
(Company(c, h) & Industry(f, c)) ==> Product(c, f)
(Company(c, h) & Industry(f, c)) ==> Export(h, f)


''',

    'queries': '''
Make(x)
ParentCompanies(x)
Parent(p, c)
Headquarters(c, h)
Product(c, a)


''',
#((Company(c, h) & Industry(a, c) & Industry(f, c)) ==> Result(c)
#Result(c)              attempt to find specifc company that exports a specific product in a specific country

#(Export(n, a) & Bikes(m))    attempt to include more than just one

}


Examples = {
    'farmer': farmer,
    'weapons': weapons,
    'vehicles': vehicles,
}

