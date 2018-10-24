Belmont = {
    'kb': '''
Algebra(Madison)
Graphing(Madison)
Integrals(Madison)
Lists(Madison)
Java(Madison)
Recursion(Madison)
History(Madison)

Algebra(Jacob)
Graphing(Jacob)
Integrals(Jacob)
Zplane(Jacob)

Algebra(Hooper)
Graphing(Hooper)
Integrals(Hooper)
Lists(Hooper)
Java(Hooper)
Recursion(Hooper)
History(Hooper)
Zplane(Hooper)

Graphing(Levi)
Graphing(Omar)


(Algebra(x) & Graphing(x)) ==> Cal1(x)
(Cal1(x) & Integrals(x)) ==> Cal2(x)
(Cal2(x) &  Zplane(x)) ==> Cal3(x)
(Lists(x) & Java(x)) ==> Pro1(x)
(Pro1(x) & Recursion(x)) ==> Pro2(x)
(Pro2(x) & History(x)) ==> Prola(x)
(Cal1(x) & Cal2(x)) ==> SmartPerson(x)
(Pro1(x) & Pro2(x))==> JavaExpert(x)
(Prola(x) & Pro2(x) & Pro1(x)) ==> Compmajor(x)
(Cal1(x) & Cal2(x) & Cal3(x)) ==> Mathmajor(x)
(SmartPerson(x) & JavaExpert(x) & Cal3(x) & Prola(x)) ==> KingofSmartness(x)
(Graphing(x)) ==> BUStudent(x)







''',

'queries':'''

Mathmajor(x)
SmartPerson(x)
JavaExpert(x)
Compmajor(x)
KingofSmartness(x)
BUStudent(x)

''',
     'Differ': '''


   ''',
'limit': 5,
}

Examples = {
    'Belmont': Belmont,

}
