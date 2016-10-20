farmer = {
    'kb': '''
Human(Tom)
Fish(Fishy)
Fish(Trout)
Trout(Rainbow)
Bird(Blue)
Fowl(Bird)
Food(Fishy, Jaws)
Food(Blue, Gator)
Friend(Rainbow, Blue)
Friend(Jaws, Gator)
Friend(Jim, Jessie)
Father(Bigblue, Blue)
Father(Bigfishy, Fishy)
Father(Jim, Tom)
Father(Jim, Jessie)
Owner(Tom, Fishy)
Owner(Tom, Rainbow)
Brother(Jessie, Tom)
Home(Cage, Blue)
Home(Bowl, Rainbow)

(Human(h) & Bird(b)) ==> Loves(h, b)
Loves(h, b) & Human(h) ==> Owner(h, b)
(Food(x, y)) ==> Eats(x, y)
(Trout(t)) ==> Fish(t)
(Home(c, h) & Bird(h)) ==> Cage(c)

''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Fish(s)
Father(u, i)
Owner(x, z)
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

Examples = {
    'farmer': farmer,
    'weapons': weapons,
}