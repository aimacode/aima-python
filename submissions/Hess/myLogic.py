house = {
    'kb': '''
Man(Eli)
Dog(Chief)
Rat(Stinky)
Bug(Slimey)
Owner(Eli, Chief)
Food(Blueberry)

Dog(d) ==> Pet(p)
(Rat(r) & Bug(b)) ==> Pest(e)

(Man(m) & Dog(d)) ==> Loves(m,d)
(Dog(d) & Man(m)) ==> Loves(d,m)
(Food(f) & Dog(d)) ==> Eats(d,f)
(Food(f) & Rat(r)) ==> Eats(r,f)
(Food(f) & Bug(b)) ==> Eats(b,f)
(Rat(r) & Dog(d)) ==> Hunts(d,r)
(Bug(b) & Rat(r)) ==> Hunts(r,b)
(Dog(d) & Rat(r)) ==> Hides(r,d)
(Rat(r) & Bug(b)) ==> Hides(b,r)

''',

    'queries':'''

Loves(x,y)
Hunts(x,y)
Hides(x,y)
Eats(x,y)

''',
}
Examples = {
    'house': house,

}