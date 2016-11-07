'''
This logic represents different types of instruments/musicians and the rules define whether or not they can play together or not.
'''



music = {
'kb': '''
    musician(Kevin)
    musician(Alice)
    musician(Kim)
    musician(Silas)
    instruments(bassist)
    instruments(drummer)
    instruments(guitarist)
    instruments(singer)
    classical(brass)
    classical(percussion)
    classical(strings)
    classical(woodwinds)
    range(alto)
    range(bass)
    range(soprano)
    range(tenor)
    rhythm(bassist, drummer)
    rhythm(drummer, guitarist)
    talent(Kim, bassist)
    talent(Alice, singer)
    talent(Kevin, guitarist)
    talent(Silas, drummer)
    harmony(alto, soprano)
    harmony(bass, tenor)
    harmony(alto, tenor)
    harmony(bass, soprano)
    harmony(strings, soprano)
    harmony(strings, tenor)

    harmony(a,b) & harmony(c,d) ==> Harmony(a,d)
    range(x) & range(y) & harmony(x,y) ==> Duet(x,y)
    classical(x) & range(y) & harmony(x,y) ==> Opera(x,y)
    rhythm(a,b) & instrument(c) ==> Band(a,b,c)
    vocalist(x) & talent(x, guitarist) ==> Solo(x)
    classical(x) & classical(y) & classical(z) ==> Orchestra(x,y,z)

''',

    'queries':'''
    Harmony(a, soprano)
    Duet(alto, y)
    Opera(strings, y)
    Band(bassist, y, singer)
    Solo(x)
    Orchestra(x,woodwinds,percussion)

''',
}

Examples = {
    'music': music,
}



