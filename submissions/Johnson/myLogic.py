bartending = {
    'kb': '''

OldEnough(Tim)
Father(Jim, Tim)
ID(TimsID, Tim)
ID(JimsID, Jim)
OrdersAlcohol(Tim)
OrdersAlcohol(Jim)
Differs(Jim, Tim)
Differs(Tim, Jim)

OldEnough(James)
Father(John, James)
ID(JamesID, James)
ID(JamesID, John)
Differs(James, John)
Differs(John, James)

OldEnough(Tyler)
OldEnough(Chris)
Father(Matthew, Tyler)
Father(Matthew, Chris)
ID(Tyler, Tyler)
ID(Tyler, Chris)
Differs(Tyler, Chris)
Differs(Chris, Tyler)
OrdersAlcohol(Matthew)

NotOldEnough(Manning)
HasWeapon(Manning)
Father(Robert, Manning)

OldEnough(Stewart)
OldEnough(Bruce)
Friends(Stewart, Bruce)
Friends(Bruce, Stewart)
OrdersAlcohol(Stewart)
OrdersWater(Bruce)

OldEnough(Larry)
OldEnough(Ryan)
Friends(Ryan, Larry)
Friends(Larry, Ryan)
OrdersAlcohol(Larry)
OrdersAlcohol(Ryan)

(OrdersAlcohol(x) & OldEnough(x)) ==> Serve(x)
(Father(x, y) & OldEnough(y)) ==> OldEnough(x)
OrdersWater(x) ==> Serve(x)
ID(x, y) & ID(x, z) & Differs(y, z) ==> KickOut(y)
Father(x, y) & KickOut(y) & KickOut(x) & OldEnough(y) ==> Shame(x)
Father(x, y) & KickOut(y) ==> Inform(y, x)
HasWeapon(x) ==> KickOut(x)
Father(x, y) & KickOut(y) & NotOldEnough(y) ==> KickOut(x)
Friends(x, y) & OrdersAlcohol(x) & Serve(x) & OrdersWater(y) & Serve(y) ==> WillDrive(y)
Friends(x, y) & OrdersAlcohol(x) & Serve(x) & OrdersAlcohol(y) & Serve(y) ==> CallTaxiFor(x)


''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Serve(x)
KickOut(x)
Inform(x, y)
Shame(x)
WillDrive(x)
CallTaxiFor(x)
''',
#    'limit': 1,
}
Examples = {
    'bartending': bartending,
}