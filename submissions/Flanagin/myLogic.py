

music = {
    'kb': '''
Instrument(Flute)
Piece(Undine, Reinecke)
Piece(Carmen, Bourne)


(Instrument(x) & Piece(w, c) & Era(c, r)) ==> Program(w)
Era(Reinecke, Romantic)
Era(Bourne, Romantic)


    ''',

    'queries': '''
    Program(x)
    ''',
}

life = {
    'kb': '''
Musician(x) ==> Stressed(x)
(Student(x) & Text(y)) ==> Stressed(x)

Musician(Heather)
    ''',

    'queries': '''
Stressed(x)
    '''

}



Examples = {
    'music': music,
    'life': life
}