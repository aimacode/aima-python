romeoAndJuliet = {
    'kb': '''
Lovers(Romeo, Juliet)
Friends(Romeo, Mercutio)
Reader(Me)
Narrator(Chorus)
Father(FatherCapulet, Juliet)
Mother(LadyMontague, Romeo)
Capulet(Tybalt)
Montague(Benvolio)
(Father(d, Juliet)) ==> Capulet(d)
(Mother(p, Romeo)) ==> Montague(p)
(Friends(Romeo, f) & Capulet(c)) ==> Despises(f, c)
(Montague(m) & Capulet(c)) ==> Despises(m, c)
(Lovers(x, y)) ==> Forbidden(x, y)
(Forbidden(x, y)) ==> Tragic(x, y)
(Reader(q)) ==> Awareofirony(q)
(Narrator(n)) ==> Awareofirony(n)
(Awareofirony(i)) ==> Frustrated(i)




''',

    'queries':'''
    Capulet(x)
    Montague(x)
    Despises(x , y)
    Tragic(x , y)
    Frustrated(x)
    

    ''',
    'limit': 10
}

Examples = {
    'Romeo and Juliet': romeoAndJuliet
}

