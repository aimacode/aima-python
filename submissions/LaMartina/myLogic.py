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

war = {
    'kb': '''
Country(USA)
Country(Iraq)
Country(GreatBritain)
Country(Iran)
Country(France)
Country(Germany)
Country(Japan)
Person(Jim)
Person(Sam)
Allies(USA,GreatBritain)
Allies(USA,France)
Allies(USA, Germany)
Allies(Iraq,Iran)
Allies(Iraq, Japan)
Allies(x,y) ==> Allies(y,x)
Allies(x,y) & Allies(y,z) ==> Allies(x,z) & Allies(z,x)
Attacks(Iraq,USA)
Attacks(Jim,Iran)
Attacks(Jim,Sam)
Person(x) & Country(y) & Attacks(x,y) ==> Dead(x)
Country(x) & Country(y) & Attacks(x,y) ==> DeclaresWar(x,y) & DeclaresWar(y,x)
DeclaresWar(x,y) ==> Attacks(x,y)
DeclaresWar(x,y) & Allies(x,z) ==> DeclaresWar(z,y)
DeclaresWar(USA,y) & DeclaresWar(Germany,y) & DeclaresWar(France,y) & DeclaresWar(GreatBritain,y) ==> Destroyed(y)


''',
    #Contains Countries who can be allies and attack eachother. When one country attacks another, war is declared and allies come
    #to help. There are also persons who can attack countries, but cannot start wars and die because of their efforts.
    #Certain groups of countries destroy others in war.
    'queries':'''
Allies(Germany,y)
Dead(x)
Destroyed(x)
DeclaresWar(x,Iran)
''',
}
Examples = {
   # 'farmer': farmer,
   # 'weapons': weapons,
    'war': war,
}
