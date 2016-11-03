LumberFarmers = {
    'kb': '''
Flannel(Jack)
Flannel(Timmy)
Overalls(Jimmy)
Overalls(Phil)
Axe(Jack)
Axe(Timmy)
Pitchfork(Jimmy)
Pitchfork(Phil)
StrawHat(Phil)
Schnitzel(Timmy)
Bodies(Timmy)
Bodies(Phil)
Bodies(Jack)
Wife(Mindy, Jack)
Wife(Mindy, Phil)

Flannel(x) & Axe(x) ==> Lumberjack(x)
Overalls(y) & Pitchfork(y) ==> Farmer(y)
Flannel(x) & Axe(x) & Schnitzel(x)==> RealLumberjack(x)
Overalls(y) & Pitchfork(y) & StrawHat(y) ==> TrueFarmer(y)
#(Lumberjack(x) | Farmer(x)) & Bodies(x) ==> CrazedMurderer(x)
Farmer(x) & Bodies(x) ==> CrazedMurderer(x)
Lumberjack(x) & Bodies(x) ==> CrazedMurderer(x)
Wife(x, y) & Wife (x, z) ==> Problems(y, z)

''',
    'queries':'''
     Lumberjack(x)
     Farmer(x)
     RealLumberjack(x)
     TrueFarmer(x)
     CrazedMurderer(x)
     Problems(y, z)

''',
    'limit': 100
}

Examples = {
    'lumberfarmers': LumberFarmers,
}