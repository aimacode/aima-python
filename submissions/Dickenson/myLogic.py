evolution = {
    'kb': '''
Kingdom(Animalia, Chordata)
 Phylum(Chordata, Reptilia)
 Phylum(Chordata, Aves)
  Class(Reptilia, Squamata)
  Class(Reptilia, Crocodilia)
   Order(Squamata, Serpentes)
   Order(Squamata,Scincomorpha)
    Suborder(Serpentes, GarterSnake)
    Suborder(Serpentes, ReticulatedPython)
    Suborder(Serpentes, BlueLippedSeaKrait)
    Suborder(Scincomorpha, CommonGreySkink)
    Suborder(Scincomorpha, RainbowSkink)
    Suborder(Scincomorpha, NorthernBlueTongueSkink)
   Order(Crocodilia, Crocodylidae)
   Order(Crocodilia, Alligatoridae)
    Suborder(Crocodylidae, SaltwaterCrocodile)
    Suborder(Crocodylidae, NileCrocodile)
    Suborder(Alligatoridae, AmericanAlligator)
  Class(Aves, Anseriformes)
   Order(Anseriformes, Anatidae)
   Order(Anseriformes, Anseranatidae)
    Suborder(Anatidae, BlackBelliedWhistlingDuck)
    Suborder(Anseranatidae, MagpieGoose)

Suborder(w, x) & Suborder(w, y) ==> EvolutionarySibling(x, y)

#Can't find a way to do ~EvolutionarySibling(p, q); that would make cousin more accurate. As it is, it also generates all siblings
#(similar to how siblings also generates itself. Same for future methods, which include every level shallower as well. Therefore, each method is
#now prefaced with "at least", because it will not only give you species at that relationship level, but also every level shallower.)

Order(w, x) & Order(w, y) & Suborder(x, q) & Suborder(y, p) ==> AtLeastEvolutionaryCousin(q, p)
Class(w, x) & Class(w, y) & Order(x, l) & Order(y, f) & Suborder(l, q) & Suborder(f, p) ==> AtLeastEvolutionaryTwiceRemoved(q, p)
Phylum(w, o) & Phylum(w, j) & Class(o, x) & Class(j, y) & Order(x, l) & Order(y, f) & Suborder(l, q) & Suborder(f, p) ==> AtLeastDistantRelatives(q, p)


''',
    'queries':'''
    EvolutionarySibling(GarterSnake, y)
    AtLeastEvolutionaryCousin(q, SaltwaterCrocodile)
    AtLeastEvolutionaryTwiceRemoved(CommonGreySkink, p)
    AtLeastDistantRelatives(MagpieGoose, p)
    EvolutionarySibling(x, y)
''',
   'limit': 100,
}

Examples = {
    'evolution': evolution,
}