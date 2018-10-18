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

zombies = {
    'kb': '''
(Zombies(z) & Virus(v) & Attack(z, v, p) & People(p)) ==> WalkingDead(p) 
Carry(Dead, RNA)
ZombieVirus(RNA) 
(WalkingDead(p) & Carry(Dead, z)) ==> Attack(Earth, z, Dead)
ZombieVirus(z) ==> Virus(z)
Inflate(p, BAD) ==> People(p)
Zombie(Earth)
Inflate(Dead, BAD)
''',
    'queries':'''
People(p)
''',
#    'limit': 1,
}


Examples = {
    'farmer': farmer,
    'weapons': weapons,
    'zombies': zombies,
}