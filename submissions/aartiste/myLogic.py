genesis = {
    'kb': '''
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
''',
    'queries':'''
    Grandfather(x, y)
    #Sibling(x, Esau)
    #Sibling(Esau, x)
    #Cousin(x, Esau)
    #Cousin(Esau, y)
''',
   'limit': 100,
}

Examples = {
    'genesis': genesis,
}