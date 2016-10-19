twilight = {
    'kb': '''
Human(Bella)
Vampire(Edward)
Werewolf(Jacob)
Dad(Charlie)
Dad(Doctor)
(Dad(d, h) & Human(h)) ==> Human(d)
(Dad(d, w) & Werewolf(w)) ==> Werewolf(d)
(Vampire(v, w) & Werewolf(w)) ==> Eats(v, w)
(Werewolf(w, v) & Vampire(v)) ==> Eats(w, v)
(Werewolf(w, h) & Human(h)) ==> Loves(w, h)
(Vampire(v, h) & Human(h)) ==> Loves(v, h)
(Vampire(v, h) & Human(h)) ==> Loves(v, h)
(Vampire(v, h) & Human(h)) ==> Hunts(v, h)
(Werewolf(w, v) & Vampire(v)) ==> Hunts(w, v)
(Dad(d, v) & Vampire(v)) ==> Hunts(d, v)
(Dad(d, w) & Werewolf(w)) ==> Hunts(d, w)
(Dad(d, h) & Human(h)) ==> Hunts(d, h)
(Human(h)) ==> EmotionallyUnstable(h)
(Vampire(v)) ==> EmotionallyUnstable(v)
(Werewolf(w)) ==> EmotionallyUnstable(w)
(Human(h, w, v) & Vampire(v) & Werewolf(w)) ==> Conflicted(h, w, v)
''',
    'queries':'''
Eats(x, y)
Loves(x, y)
Conflicted(x, y, z)
Hunts(x, y)
EmotionallyUnstable(x)
''',
}

Examples = {
    'twilight': twilight,
}