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
####
language = {
    'kb': '''

AMan(Adam)
AWoman(Becky)
CMan(Chandler)
CWoman(Diane)
MMan(Eugene)
MWoman(Fiona)

Native(American, English)
Native(Canadian, English)
Native(Mexican, Spanish)

Second(American, English, Spanish)
Second(Canadian, English, French)
Second(Mexican, Spanish, English)

(AMan(x) & AWoman(y)) ==> American(x,y)
(CMan(x) & CWoman(y)) ==> Canadian(x,y)
(MMan(x) & MWoman(y)) ==> Mexican(x,y)

(AMan(x)) ==> Educated(x)
(CMan(x)) ==> Educated(x)
(MMan(x)) ==> Educated(x)

(American(a,b) & Educated(x)) ==> SmartA(Adam)
(SmartA(Adam)) ==> ASecond(x)
(American(a,b)) ==> ANative(American, English)

(Canadian(c,d) & Educated(x)) ==> SmartC(Chandler)
(SmartC(Chandler)) ==> CSecond(x)
(Canadian(c,d)) ==> CNative(x)

(Mexican(e,f) & Educated(x)) ==> SmartM(Eugene)
(SmartM(Eugene)) ==> MSecond(x)
(Mexican(e,f)) ==> MNative(x)

(American(a,b)) ==> Native(am,en)
(Canadian(c,d)) ==> Native(ca,en)
(Mexican(e,f)) ==> Native(m,s)

(American(a,b) & SmartA(a)) ==> Second(am,s)
(Canadian(c,d) & SmartC(c)) ==> Second(ca,fr)
(Mexican(e,f) & SmartM(e)) ==> Second(m,en)

''',

    'queries':'''
American(a,b)
Canadian(c,d)
Mexican(e,f)
Educated(x)
Native(x,y)
Second(x,y,z)
''',
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

Examples = {
  #  'farmer': farmer,
    'language': language,
  #  'weapons': weapons,
}