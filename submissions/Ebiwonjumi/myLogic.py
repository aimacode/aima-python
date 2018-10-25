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

nigerians = {
      'kb': '''
Igbo(Chukwu)
Igbo(Kamsi)
Igbo(Ifeanyi)
Igbo(Prince)
Yoruba(Ayo)
Yoruba(Ope)
Yoruba(Tosin)
Yoruba(Tope)
Hausa(Suleiman)
Hausa(Magashi)
Hausa(Baaki)
Hausa(Mubarak)
Hausa(Abdul)
American(Jake)
American(Josh)
American(Jessica)
American(Kathryn)
American(Sam)
Ghanian(Kotu)
Ghanian(Chuwa)
Ghanian(Kino)
Alien(jg)
 (Igbo(i)) ==> Rich(i)
 (Hausa(h)) ==> Farmer(h)
 (Yoruba(y)) ==> Smart(y)
 (Alien(l))  ==> NotHuman(l)
 (Hausa(h) & Yoruba(y)) ==> Friends(h, y)
 (Igbo(i)) ==> Nigerian(i)
 (Hausa(h)) ==> Nigerian(h)
 (Yoruba(y)) ==> Nigerian(y)
 (Nigerian(n)) ==> African(n)
 (Ghanian(g)) ==> African(g)
 (Nigerian(n) & Ghanian(g)) ==> Neighbors(n, g)
 (African(a)) ==> Human(a)
 (African(a) & American(m)) ==> Enemies(a, m)
 (NotHuman(j) & Human(h)) ==> NewWaters(h, j)
 
 
''',
    'queries':'''
Rich(x)
Friends(x, y)
Nigerian(n)
African(a)
NotHuman(z)
Human(h)
Enemies(j, x)
NewWaters(f, k)
''',

}

Examples = {
   # 'farmer': farmer,
   # 'weapons': weapons,
   'nigerians': nigerians,
}
