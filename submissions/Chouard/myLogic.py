wizards = {
    'kb': '''
    ((Parent(y, x) & Muggle(y)) & (Parent(z, x) & Muggle(z))) ==> MudBlood(x)
    ((Parent(y, x) & Wizard(y)) & (Parent(z, x) & Wizard(z))) ==> PureBlood(x)
    PureBlood(x) ==> Prejudiced(x)
    PureBlood(x) & Muggle(y) ==> Hates(x, y)
    Magic(x) ==> Wizard(x)
    Muggle(x) & Wizard(y) ==> Oblivious(x, y)
    PureBlood(x) & MudBlood(y) ==> Bullies(x, y)
    Magic(Lucius)
    Magic(Narcissa)
    Parent(Lucius, Draco)
    Parent(Narcissa, Draco)
    Muggle(MrGranger)
    Muggle(MsGranger)
    Parent(MrGranger, Hermione)
    Parent(MsGranger, Hermione)
    ''',
    'queries': '''
    Oblivious(x, y)
    PureBlood(x)
    Prejudiced(x)
    Hates(x, y)
    Bullies(x, y)
    ''',
    'limit': 1
}

Examples = {
    'wizards': wizards
}
