# common utilities for grading
from utils import isnumber

roster = [
    'Anderson','Ban','Becker',
    'Blue','Capps','Conklin',
    'Dickenson','Fritz','Haller',
    'Hawley','Hess','Johnson',
    'Karman','Kinley','LaMartina',
    'McLean','Miles','Ottenlips',
    'Porter','Sery','VanderKallen',
    'aardvark',
    'aartiste',
    'zzzsolutions',
]

def print_table(table, header=None, sep='   ',numfmt='%g',
                njust='rjust', tjust='ljust'):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first rows.
    sep is the separator between columns, e.g. '|' or ', '
    numfmt is the format for all numbers; you might want e.g. '%6.2f'.
    (If you want different formats in different columns,
    don't use print_table.)
    njust and tjust justify the numbers and text, e.g. 'center'
    """
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