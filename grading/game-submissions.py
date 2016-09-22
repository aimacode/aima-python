import importlib
import traceback
from utils import isnumber
from math import inf
from games import alphabeta_search

class MyException(Exception):
    pass

roster = ['Anderson', 'Ban','Becker','Blue','Capps','Conklin','Dickenson',
          'Fritz','Haller','Hawley','Hess','Johnson','Karman','Kinley',
          'LaMartina','McLean','Miles','Ottenlips','Porter','Sery',
          'VanderKallen',
          'aardvark','zzzsolutions',
          ]

def play_game(game, *players, verbose=False):
    """Play an n-person, move-alternating game."""

    state = game.initial
    if verbose:
        print('Initial State:')
        game.display(state)
    while True:
        for player in players:
            move = player(game, state)
            state = game.result(state, move)
            if verbose:
                print(player.label, 'chooses', str(move))
                game.display(state)
            if game.terminal_test(state):
                if not verbose:
                    game.display(state)
                return game.utility(state, game.to_move(game.initial))

def print_table(table, header=None, sep='   ', numfmt='%g', njust='rjust', tjust='ljust'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row.
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in different columns,
    don't use print_table.) sep is the separator between columns."""
    if len(table) == 0:
        return
    justs = [njust if isnumber(x) else tjust for x in table[0]]

    if header:
        r = 0
        for row in header:
            table.insert(r, row)
            r += 1

    table = [[(numfmt % x) if isnumber(x) else x for x in row]
             for row in table]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in table]))))

    for row in table:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))

class TiredOfPlaying(Exception):
    pass

def make_ab(depth):
    function = lambda game, state: alphabeta_search(state, game, d=depth)
    function.label = "AB(" + str(depth) + ")"
    return function

def printActions(actions):
    message = "Valid moves:"
    for i in range(len(actions)):
        message += ' ' + str(i + 1) + ':' + str(actions[i])
    print(message)

def query_player(game, state):
    "Make a move by querying standard input."
    actions = game.actions(state)
    printActions(actions)
    success=False
    while(not success):
        move_string = input('Your move? ').strip()
        if move_string == 'quit':
            raise TiredOfPlaying()
        try:
            index = eval(move_string)
            assert index in range(1,len(actions)+1)
            success=True
        except:
            print('"' + move_string + '" is not a valid index.')
            success = False
    move = actions[index - 1]
    return move

query_player.label = "Ask"

def try_to_play(game):
    done = False
    while not done:
        dstring = input('Level of Opponent (0..' + str(len(AB_searches)-1) + ', s to skip): ')
        if dstring.strip().lower()[0] == 's':
            break
        try:
            depth = int(dstring)
            assert depth >= 0
        except:
            print('"' + dstring + '" is not a positive integer.')
            continue
        gstring = input('Go first? [yN]: ')
        goFirst = gstring.lower().strip() == 'y'
        try:
            ab_player = make_ab(depth)
            if goFirst:
                first = query_player
                second = ab_player
            else:
                first = ab_player
                second = query_player
            score = play_game(game, first, second, verbose=True)
            print(first.label + ' wins.' if score > 0 else
                  second.label + ' wins.' if score < 0 else
                  'A Tie.')
        except TiredOfPlaying:
            pass
        except:
            traceback.print_exc()
            done = True

AB_searches = [ make_ab(d) for d in range(9)]

def try_games(games):
    for g in games:
        topLeft = str(g)[1:-1]
        header = [[str(topLeft)]]
        maxChars = len(topLeft)
        for i in range(len(AB_searches)):
            header[0].append('AB(' + str(i) + ')')
        table = []
        for state in games[g]:
            column = [str(state)[:maxChars]]
            for abSearch in AB_searches:
                bestMove = abSearch(g, state)
                column.append(str(bestMove))
            table.append(column)
        print_table(table, header, tjust='rjust')
        try_to_play(g)

submissions = {}
scores = {}

message1 = 'Submissions that compile:'
for student in roster:
    try:
        # http://stackoverflow.com/a/17136796/2619926
        mod = importlib.import_module('submissions.' + student + '.mygames')
        submissions[student] = mod.myGames
        message1 += ' ' + student
    except ImportError:
        pass
    except:
        traceback.print_exc()

print(message1)
print('----------------------------------------')

for student in roster:
    if not student in submissions.keys():
        continue
    scores[student] = []
    try:
        games = submissions[student]
        print('Games from:', student)
        try_games(games)
    except:
        traceback.print_exc()

    print(student + ' scores ' + str(scores[student]) + ' = ' + str(sum(scores[student])))
    print('----------------------------------------')