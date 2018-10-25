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

FoodChain = {
    'Differ': ''' 
Amber 
Pete 
John
Jerry
Falchion
Skippy
Lisa
Grup
Debbie
Jax 
Lassie
     

''',

    'kb': '''
Fox(Amber)
Fox(Jax)
Brother(Jax, Amber)
Sister(Amber,Jax)
Rabbit(Pete)
Hare(Lassie)
Lion(Steve)
Owl(John)
Mouse(Jerry)
Bird(Falchion)
Grasshopper(Skippy)
Carrot(Lisa)
Grain(Grup)
Grass(Debbie)
DartFrog(Guppy)
Mammal(Amber)
Mammal(Jax)
Mammal(Steve)
Bird(John)
Bird(Falchion)

Herbivore(Lassie)
Herbivore(Jerry)
Herbivore(Skippy)
Herbivore(Pete)
Plants(Grain)
Plants(Grass)
Plants(Carrot)
Poisonous(Guppy)
(Owl(x)) ==> Nocturnal(x)
(Fox(x)) ==> Nimble(x)
(Mammal(f)) & Poisonous(o) ==> Despises(f,o)
(Bird(b)) & Poisonous(o) ==> Despises(b,o)
(Mouse(x)) & Rabbit(d) & Hare(t) ==> Rodents(x,d,t)
(Nocturnal(n)) & Rodents(x,d,t) ==> Hunts(n,x,d,t)
(Hunts(n,x,d,t)) & Bird(n) ==> Apex(n)

(Despises(f,o)) & Rodents(x,d,t) ==> Devours(f,x,d,t)
(Apex(f)) & Herbivore(d) ==> TerrorizesAtNight(d,f)
(Plants(f)) & Herbivore(d) ==> Fears(f,d)
(Herbivore(h)) & Plants(p) ==> Eats(h,p)
(Despises(f,d)) & Eats(h,p) ==> DoesntEat(f,p)
(Bird(b)) & Mammal(f) ==> Carnivores(b,f)
(Brother(b , s)) ==> Siblings(b,s) 

''',

    'queries':'''
    
    
    Despises(Falchion,y)
    Despises(x,y)
    Fox(x)
    Nimble(x)
    Apex(x)
    Eats(x,y)
    Siblings(f,s)
    Fears(x,y)
    Fears(Carrot, y)
    Rodents(x,y,z)
    Hunts(x,y,z,q)
    TerrorizesAtNight(x,y)
    Devours(x,y,z,r)
    DoesntEat(x,y)
    Carnivores(b,f)
    Nocturnal(x)
    ''',
'limit': 20,
}


Examples = {
    'farmer': farmer,
    'weapons': weapons,
    'FoodChain': FoodChain
}
