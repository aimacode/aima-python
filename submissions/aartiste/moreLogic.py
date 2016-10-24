'''
Father(Abraham, Isaac)
 Father(Isaac, Esau)
  Father(Esau, Eliphaz)
  Father(Esau, Jeush)
 Father(Isaac, Jacob)
  Father(Jacob, Reuben)
  Father(Jacob, Simeon)
  Father(Jacob, Levi)
Father(Abraham, Ishmael)
 Father(Ishmael, Nebaioth)
 Father(Ishmael, Kedar)

Father(w, x) & Father(x, y) ==> Grandfather(w, y)
Father(w, x) & Father(w, y) ==> Sibling(x, y)
Father(w, x) & Father(y, z) & Sibling(w, y) ==> Cousin(x, z)

#Differ
Father(w, x) & Father(w, y) & Differ(x, y) ==> Sibling(x, y)

Differ(Isaac, Ishmael)
Differ(Esau, Jacob)
Differ(Reuben, Simeon)
Differ(Reuben, Levi)
Differ(Simeon, Levi)
Differ(Nebaioth, Kedar)

Differ(x, y) ==> Differ(y, x)
#Differ(x, y) ==> Differ(y, x)

Differ(Ishmael, Isaac)
Differ(Jacob, Esau)
Differ(Simeon, Reuben)
Differ(Levi, Reuben)
Differ(Levi, Simeon)
Differ(Kedar, Nebaioth)
'''