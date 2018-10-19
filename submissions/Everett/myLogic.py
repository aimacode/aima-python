family = {
    'kb': '''
Daughter(Sandy)
Daughter(Dale)
Cousin(Jasmine)
Daughter(Madison)
Daughter(Megan)
Daughter(Morgan)
Mother(Mattie, Sandy)
Mother(Mattie,Dale)
Mother(Dale,Jasmine)
Mother(Sandy,Madison)
Mother(Sandy,Morgan)
Mother(Sandy,Megan)
(Mother(x,y) & Daughter(y)) ==> Sister(y)
(Sister(y) & Daughter(y)) ==> Girl(y)
(Mother(y,z)) ==> Girl(y)
(Mother(y,z) & Sister(y) & Girl(y)) ==> Aunt(y)



''',

'queries':'''

Sister(y)
Girl(x)
Daughter(x)
Aunt(x)

''',
     'Differ': '''
"Sandy", "Dale"

   ''',



'limit': 4,


}

Examples = {
    'Family': family,

}
