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

musicians = {
    'kb': '''
    Guitarist(JimmyPage)
    Drummer(JohnBonham)
    Drummer(LarryMullen)
    Singer(RobertPlant)
    Singer(JimiHendrix)
    Singer(Bono)
    Guitarist(JimiHendrix)
    Guitarist(TheEdge)
    Bassist(AdamClayton)
    Bassist(JohnPaulJones)
    Guitarist(x) ==> Musician(x)
    Drummer(x) ==> Musician(x)
    Bassist(x) ==> Musician(x)
    Singer(x) ==> Musician(x)
    Band(JimmyPage, RobertPlant, JohnBonham, JonhPaulJones)
    Band(Bono, TheEdge, LarryMullen, AdamClayton)
    Band(JimiHendrix)
    British(Bono)
    British(TheEdge)
    British(AdamClayton)
    British(LarryMullen)
    British(RobertPlant)
    British(JimmyPage)
    British(JohnBonham)
    British(JohnPaulJones)
    (British(x) & Musician(x)) ==> BritishMusician(x)
    (Guitarist(x) & Drummer(y) & Bassist(z) & Singer(w)) ==> FullBand(x,y,z,w)
    ''',
    'queries': '''
    Musician(x)
    Guitarist(x)
    Singer(x)
    Bassist(x)
    Drummer(x)
    BritishMusician(x)
    Band(x,y,z,w)
    FullBand(x,y,z,w)
    ''',
    'limit': 10
}


Examples = {
    # 'farmer': farmer,
    # 'weapons': weapons,
    'musicians': musicians
}
