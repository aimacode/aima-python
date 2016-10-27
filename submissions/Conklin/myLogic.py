heroesAndVillains = {
     'kb': '''
 Protagonist(Geralt)
 Protagonist(Tidus)
 Protagonist(Dragonborn)
 Protagonist(Dante)
 Antagonist(Eredin)
 Antagonist(Jecht)
 Antagonist(Alduin)
 Antagonist(Vergil)
 mortalEnemies(Geralt, Eredin)
 mortalEnemies(Tidus, Jecht)
 mortalEnemies(Dragonborn, Alduin)
 mortalEnemies(Dante, Vergil)


 (Protagonist(p) & Antagonist(a)) ==> Conflict(p, a)
 mortalEnemies(p,a) & Antagonist(a) ==> Protagonist(p))
 Protagonist(p) & Protagonist(q) ==> goodAlliance(p,q)
 Antagonist(a) & Antagonist(z) ==> badAlliance(a,z)




 ''',

     'queries':'''
 Protagonist(p)
 Antagonist(a)
 mortalEnemies(p,a)
 goodAlliance(p,q)
 badAlliance(a,z)
 ''',
#CHANGE
 }



 Examples = {
     'heroesAndVillains': heroesAndVillains,
 }