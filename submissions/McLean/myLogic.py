cats = {
    'kb': '''



Name(Felis, Catus)
Name2(Catus, Cat)
Name(Felis, Chaus)
Name2(Chaus, JungleCat)
Name(Panthera, Tigris)
Name2(Tigris, Tiger)
Name(Panthera, Onca)
Name2(Onca, Jaguar)
Name(Panthera, Pardus)
Name2(Pardus, Leopared)
Name(Panthera, Leo)
Name2(Leo, Lion)
Name(Felinae, Lynx)
Name2(Lynx, Lynx)
Name(Acinoyx, Jubatus)
Name2(Jubatus, Cheetah)
Classification(Felidae, Felis)
Classification(Felidae, Panthera)
Classification(Felidae, Felinae)
Classification(Felidae, Acinoyx)
Family(Felidae)



Name(w, x) & Name(x, y) ==> Related(x, y)
Name(w, x) & Name2(x, y) ==> Called(y)
Name(w, x) & Classification(y, w) ==> SameFamily1(y)
Related(w, x) & Related(w, y) ==> SameFamily(x, y)
Name(w, x) & Family(m) ==> TheGenus(w)
Name(g, h) & Family(t) ==> TheSpecies(h)
Name(t, e) & Name(a, s) & SameFamily1(w) ==> TheFamily(w)

''',


    'queries':'''
Name(u, i)
Called(c)
TheGenus(j)
TheSpecies(z)
TheFamily(t)

''',
#    'limit': 1,
}



Examples = {
    'cats': cats,

}