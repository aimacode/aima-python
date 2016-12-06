# common utilities for grading
from utils import isnumber

roster = [
    # 'Anderson','Ban','Becker',
    # 'Blue','Capps','Conklin',
    # 'Dickenson','Fritz','Haller',
    # 'Hawley','Hess','Johnson',
    # 'Karman',
    'Kinley',
    # 'LaMartina',
    # 'McLean','Miles','Ottenlips',
    # 'Porter','Sery','VanderKallen',
    # 'aardvark',
    # 'aartiste',
    # 'zzzsolutions',
]

def print_table(table, header=[], leftColumn=[], topLeft=[],
                sep='   ', numfmt='%g', njust='rjust', tjust='ljust'):
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


    pTable = []
    if len(header) > 0:
        r = 0
        for row in header:
            pTable.insert(r, row)
            r += 1

    pTable.extend([[(numfmt % x) if isnumber(x) else x for x in row]
                   for row in table])

    if len(leftColumn) > 0:
        #justs.insert(0, njust if isnumber(leftColumn[0]) else tjust)
        pLeft = []
        if header:
            hr = 0
            for h in header:
                topLeft.append(' ')
                pLeft.insert(0, topLeft[hr])
                hr += 1
        for cell in leftColumn:
            pLeft.append(cell)
        r = 0
        for row in pTable:
            row.insert(0, pLeft[r])
            r += 1

    justs = [njust if isnumber(x) else tjust for x in pTable[len(header)]]

    sizes = list(
            map(lambda seq: max(map(len, seq)),
                list(zip(*[map(str, row) for row in pTable]))))

    for row in pTable:
        print(sep.join(getattr(
            str(x), j)(size) for (j, size, x) in zip(justs, sizes, row)))