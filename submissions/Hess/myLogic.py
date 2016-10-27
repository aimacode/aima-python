house = {
    'kb': '''
Man(Eli)
Woman(Eve)
Dog(Chief)
Rat(Stinky)
Bird(Hawky)
Bug(Slimey)
Owner(Eli, Chief)
Owner(Eve, Chief)
Owner(Eli, Hawky)
Owner(Eve, Hawky)
Pet(Chief, Eli)
Pet(Chief, Eve)
Pet(Hawky, Eli)
Pet(Hawky, Eve)
Pest(Slimey, Eli)
Pest(Slimey, Eve)
Predator(Chief, Stinky)
Predator(Hawky, Stinky)
Predator(Stinky, Slimey)
Prey(Stinky, Chief)
Prey(Stinky, Hawky)
Prey(Slimey, Stinky)
Food(Blueberry)
Scraps(Crumb)

(Man(m) & Woman(w)) ==> Loves(m,w)
(Woman(w) & Man(m)) ==> Loves(w,m)
(Man(m) & Dog(d)) ==> Loves(m,d)
(Woman(w) & Dog(d)) ==> Loves(w,d)
(Dog(d) & Man(m)) ==> Loves(d,m)
(Dog(d) & Woman(w)) ==> Loves(d,w)
(Man(m) & Bird(k)) ==> Loves(m,k)
(Woman(w) & Bird(k)) ==> Loves(w,k)
(Food(f) & Dog(d)) ==> Eats(d,f)
(Food(f) & Bird(k)) ==> Eats(k,f)
(Scraps(s) & Rat(r)) ==> Eats(r,s)
(Scraps(s) & Bug(b)) ==> Eats(b,s)
(Rat(r) & Dog(d)) ==> Hunts(d,r)
(Rat(r) & Bird(k)) ==> Hunts(k,r)
(Bug(b) & Rat(r)) ==> Hunts(r,b)
(Dog(d) & Rat(r)) ==> Hides(r,d)
(Bird(k) & Rat(r)) ==> Hides(r,k)
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