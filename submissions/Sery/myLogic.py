animals = {
    'kb': '''
        Animal(Tiger)
        Predator(Tiger)
        Predator(Lion)
        Predator(Shark)
        Prey(Chicken)
        Prey(Deer)
        Prey(Fish)
        Hunts(Tiger, Chicken)
        Hunts(Tiger, Deer)
        Hunts(Tiger, Fish)
        Hunts(Lion, Chicken)
        Hunts(Lion, Deer)
        Hunts(Lion, Fish)
        Hunts(Shark, Chicken)
        Hunts(Shark, Deer)
        Hunts(Shark, Fish)

        (Predator(b) & Prey(m)) ==> Hunts(b, m)
        (Prey(m)) ==> Runs(m)
        (Prey(m)) ==> Animal(m)
        (Predator(m)) ==> Animal(m)
        (Hunts(m, c)) ==> Eats(m, c)
        (Hunts(r, c) & Predator(r)) ==> Eats(r, c)

    ''',
    # Note that this order of conjuncts
    # would result in infinite recursion:
    # '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries': '''
        Predator(x)
        Hunts(x, y)
    ''',
    #    'limit': 1,
}


Examples = {
    'animals': animals,
}
