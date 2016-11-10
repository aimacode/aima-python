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
Country(GreatBritain)
Country(Iran)
Country(France)
Country(Germany)
Country(Iraq)
Person(Jim)
Person(Sam)
Person(Carl)
Allies(USA,GreatBritain)
Allies(USA,France)
Allies(USA,Germany)
Attacks(Jim,Iran)
Attacks(Carl,Sam)
Attacks(Iran,France)
Attacks(Iraq,Germany)
Region(MiddleEast)
Contains(MiddleEast,Iran,Iraq)
Person(x) & Country(y) & Attacks(x,y) ==> Dead(x)
Person(x) & Person(y) & Attacks(x,y) ==> Dead(y)
Attacks(x,y) & Country(x) & Country(y) ==> DeclaresWar(y,x)
DeclaresWar(x,y) & Allies(USA,x) ==> Destroyed(y)
Region(x) & Contains(x,y,z) & Destroyed(y) & Destroyed(z) ==> Sad(x)
Mother(Lucy,Sam)
Mother(Linda,Carl)
Mother(x,y) & Dead(y) ==> Sad(x)



''',
    #Contains Countries who can be allies and attack eachother. When one country attacks another, war is declared.
    # If the USA is an ally of an attacked country, then the attacker is destroyed.
    # There are also persons who can attack countries, but cannot start wars and die because of their efforts.
    # Persons can kill other persons as well. Persons also have mothers who are sad if their
    # child is dead. Regions contain countries and are sad if all of their countries are destroyed.

    'queries':'''
Dead(x)
Destroyed(x)
DeclaresWar(Germany, x)
Sad(x)

''',

}
Examples = {
   # 'farmer': farmer,
   # 'weapons': weapons,
    'war': war,
}
#DeclaresWar(x,y) & ~(Attacks(x,y)) ==> Attacks(x,y)
# Allies(x,y)  ==> Allies(y,x)
# Allies(x,y) & Allies(y,z)  ==> Allies(x,z)
