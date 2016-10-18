farmer = {
    'kb': '''
Father(Abraham, Isaac)
 Father(Isaac, Esau)
 Father(Isaac, Jacob)
  Father(Jacob, Reuben)
  Father(Jacob, Simeon)
  Father(Jacob, Levi)
  Father(Jacob, Judah)
Father(Abraham, Ishmael)
 Father(Ishmael, Nebaioth)
 Father(Ishmael, Kedar)

Father(w, x) & Father(w, y) ==> Sibling(x, y)
Father(w, x) & Father(x, y) ==> Grandfather(w, y)
Grandfather(w, x) & Grandfather(w, y) ==> Cousin(x, y)
''',
# Note that this order of conjuncts
# would result in infinite recursion:
# '(Human(h) & Mother(m, h)) ==> Human(m)'
    'queries':'''
''',
#    'limit': 1,
}

Examples = {
    'farmer': farmer,
}