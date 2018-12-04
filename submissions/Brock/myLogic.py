# farmer = {
#     'kb': '''
# Farmer(Mac)
# Rabbit(Pete)
# Mother(MrsMac, Mac)
# Mother(MrsRabbit, Pete)
# (Rabbit(r) & Farmer(f)) ==> Hates(f, r)
# (Mother(m, c)) ==> Loves(m, c)
# (Mother(m, r) & Rabbit(r)) ==> Rabbit(m)
# (Farmer(f)) ==> Human(f)
# (Mother(m, h) & Human(h)) ==> Human(m)
# ''',
# # Note that this order of conjuncts
# # would result in infinite recursion:
# # '(Human(h) & Mother(m, h)) ==> Human(m)'
#     'queries':'''
# Human(x)
# Hates(x, y)
# ''',
# #    'limit': 1,
# }
#
# weapons = {
#     'kb': '''
# (American(x) & Weapon(y) & Sells(x, y, z) & Hostile(z)) ==> Criminal(x)
# Owns(Nono, M1)
# Missile(M1)
# (Missile(x) & Owns(Nono, x)) ==> Sells(West, x, Nono)
# Missile(x) ==> Weapon(x)
# Enemy(x, America) ==> Hostile(x)
# American(West)
# Enemy(Nono, America)
# ''',
#     'queries':'''
# Criminal(x)
# ''',
# }

food = {
    'Differ': "James,Jim,Jeff,Jess, Wheat, Dairy, Corn, Bread, Pizza, Bridge, Lead",
    'kb': '''
Allergic(James, Wheat)
Allergic(James, Dairy)
Allergic(Jim, Dairy)
Allergic(Jeff, Wheat)
Eating(Jim, Bread)
Eating(James, Corn)
Eating(Jess, Pizza)
Eating(Jeff, Bread)
Contains(Pizza, Wheat)
Contains(Pizza, Dairy)
Contains(Bread, Wheat)
Contains(Bridge, Lead)
Contains(Cucumber, Water)
Contains(Corn, Water)


Eating(x, y) ==> Hungry(x, y)
(Hungry(x, y) & Contains(y, z) & Allergic(x, z)) ==> Dying(x)
(Contains(x, y) & Hungry(q, x)) ==> Food(x)
(Food(x) & Contains(x, y) & Allergic(q, y)) ==> AllergicTo(q, x)
(AllergicTo(x, y) & AllergicTo(z, y)) ==> Friends(x, z)
(Dying(x) & Friends(y, x)) ==> Funeral(x, y)


Eating(x, y) ==> Food(y)





''',
    'queries':'''
Dying(x)
Food(x)
Hungry(x, y)
AllergicTo(x,y)
Funeral(x, y)
Friends(x, y)
''',
    'limit': 100,
}

Examples = {
    #'farmer': farmer,
    #'weapons': weapons,
    'food' : food,
}
