heroesAndVillains = {
    'kb': '''
Hero(Batman)
Hero(WonderWoman)
Hero(Flash)
Hero(Superman)
Hero(GreenLantern)
Villain(LexLuther)
Villain(DoomsDay)
Villain(Penguin)
Villain(Joker)
Villain(Sinestro)
Villain(Zoom)
Sidekick(Robin, Batman)
SideKick(PieFace, GreenLantern)
Sidekick(CommisionerGordon, Batman)
Sidekick(HarleyQuinn, Joker)
Sidekick(JimmyOlsen, Superman)
Sidekick(KidFlash, Flash)
Nemesis(Batman, Joker)
Nemesis(Superman, LexLuther)
Nemesis(GreenLantern, Sinestro)
Nemesis(Flash, Zoom)


(Hero(h) & Villain(v)) ==> BeatsUp(h, v)
(Sidekick(s, h)) ==> Protects(s, h)
(Hero(h) & Sidekick(s, h)) ==> Hero(s)
Nemesis(h,v) & Villain(v) ==> Hero(h)
Nemesis(h,v) & Sidekick(s, h) & Sidekick(x,v) ==> Nemesis(s,x)
Hero(h) & Hero(y) ==> Allies(h,y)
Villain(v) & Villain(x) ==> Allies(v,x)




''',

    'queries':'''
Hero(h)
Villain(v)
Nemesis(h,v)
Allies(h,y)
''',

}



Examples = {
    'heroesAndVillains': heroesAndVillains,
}