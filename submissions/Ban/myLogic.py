twilight = {
    'kb': '''
Human(Bell)
Vampire(Ed)
Werewolf(Jake)
Sheriff(Char)

(Sheriff(s) & Human(h)) ==> Father(s, h)

(Vampire(v) & Werewolf(w)) ==> Eats(v, w)
(Werewolf(w) & Vampire(v)) ==> Eats(w, v)

(Human(h)) ==> EmotionallyUnstable(h)
(Vampire(v)) ==> EmotionallyUnstable(v)
(Werewolf(w)) ==> EmotionallyUnstable(w)

(Vampire(v) & Human(h)) ==> Loves(v, h)
(Werewolf(w) & Human(h)) ==> Loves(w, h)
(Human(h) & Vampire(v)) ==> Loves(h, v)

(Human(h) & Werewolf(w)) ==> Hunts(w, h)
(Werewolf(w) & Vampire(v)) ==> Hunts(w, v)
(Father(s, h) & Loves(h, y)) ==> Hunts(s, y)

(Human(h) & Vampire(v) & Werewolf(w) & Loves(h, v) & Loves (h, w)) ==> Conflicted(h)
''',
    'queries':'''
Human(x)
Vampire(x)
Werewolf(x)
Sheriff(x)
EmotionallyUnstable(x)
Father(x, y)
Eats(x, y)
Loves(x, y)
Conflicted(x)
Hunts(x, y)
''',
}

Examples = {
    'twilight': twilight,
}