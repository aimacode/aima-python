grant = {
    'kb': '''
Captain(Jack)
FirstMate(Bart)
Helmsmen(Sebastian)
DeckSwabber(Patches)
DeckSwabber(Barney)
DeckSwabber(Peggy)
DeckSwabber(Hans)
DeckSwabber(Red)
DeckSwabber(Bucky)
DeckSwabber(Hook)

Commodore(Nottingham)
Lieutenant(Washington)
Quartermaster(Hamilton)
Sailor(Nelson)
Sailor(Oliver)
Sailor(Johnny)
Sailor(Hilbert)
Sailor(Johnson)
Sailor(Peters)
Sailor(Edwards)

Civilian(Tom)
Kills(Tyson, Tom)

(Helmsmen(x) & DeckSwabber(y)) ==> Orders(x,y)
(FirstMate(x) & Helmsmen(y)) ==> Orders(x,y)
(Captain(x) & FirstMate(y)) ==> Orders(x,y)

(Commodore(x) & Lieutenant(y)) ==> Orders(x,y)
(Lieutenant(x) & Quartermaster(y)) ==> Orders(x,y)
(Quartermaster(x) & Sailor(y)) ==> Orders(x,y)

(Orders(x,y) & Orders(y,z)) ==> Ignores(x,z)
(Ignores(x,y) & Orders(y,z)) ==> Abuses(x,z)

Captain(x) ==> Pirate(x)
FirstMate(x) ==> Pirate(x)
Helmsmen(x) ==> Pirate(x)
DeckSwabber(x) ==> Pirate(x)
Kills(x,y) ==> Pirate(x)

Commodore(x) ==> Navy(x)
Lieutenant(x) ==> Navy(x)
Quartermaster(x) ==> Navy(x)
Sailor(x) ==> Navy(x)

(Pirate(x) & Navy(y)) ==> Hate(x,y)



''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
Orders(Hamilton,y)
Ignores(Bart,y)
Abuses(x,Bucky)
Pirate(x)


''',
    'limit' : 20,
}
Examples = {
    'grant': grant,
}
