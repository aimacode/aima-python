import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if len(program.allowed)==0:
                program.allowed = ['Left', 'Right', 'Down', 'Up']
            if bump == 'None':
                action = program.allowed[0]

                # Record coordinate change
                if lastAction == 'Left':
                    program.currX -= 1
                if lastAction == 'Right':
                    program.currX += 1
                if lastAction == 'Up':
                    program.currY += 1
                if lastAction == 'Down':
                    program.currY -= 1
            else:
                if lastAction in program.allowed:
                    program.allowed.remove(lastAction)

                #Set coordinate edges
                if lastAction == 'Left':
                    program.limits[0] = program.currX
                if lastAction == 'Right':
                    program.limits[1] = program.currX
                if lastAction == 'Down':
                    program.limits[2] = program.currY
                if lastAction == 'Up':
                    program.limits[3] = program.currY

                #Find out if the row or coll is already cleaned
                if 'Left' not in program.allowed and 'Right' not in program.allowed:
                    program.cleanedRows.append(program.currY)
                if 'Up' not in program.allowed and 'Down' not in program.allowed:
                    program.cleanedColls.append(program.currX)
                #Fill the empty list
                if len(program.allowed) == 0:
                    program.allowed = ['Left', 'Right', 'Down', 'Up']
                    program.allowed.remove(lastAction)

                #Just passing through...
                if program.allowed[0] == 'Left' and program.currX-1 in program.cleanedColls:
                    if 'Up' in program.allowed:
                        program.allowed.remove('Up')
                    if 'Down' in program.allowed:
                        program.allowed.remove('Down')
                    if len(program.allowed) == 0:
                        program.allowed = ['Left', 'Right']
                if program.allowed[0] == 'Right' and program.currX + 1 in program.cleanedColls:
                    if 'Up' in program.allowed:
                        program.allowed.remove('Up')
                    if 'Down' in program.allowed:
                        program.allowed.remove('Down')
                    if len(program.allowed) == 0:
                        program.allowed = ['Left', 'Right']
                if program.allowed[0] == 'Down' and program.currY - 1 in program.cleanedRows:
                    if 'Left' in program.allowed:
                        program.allowed.remove('Left')
                    if 'Right' in program.allowed:
                        program.allowed.remove('Right')
                    if len(program.allowed) == 0:
                        program.allowed = ['Down', 'Up']
                if program.allowed[0] == 'Up' and program.currY + 1 in program.cleanedRows:
                    if 'Left' in program.allowed:
                        program.allowed.remove('Left')
                    if 'Right' in program.allowed:
                        program.allowed.remove('Right')
                    if len(program.allowed) == 0:
                        program.allowed = ['Down', 'Up']

                #Don't go to a edge row
                if program.currY+1 == program.limits[3]:
                    if 'Up' in program.allowed:
                        program.allowed.remove('Up')
                if program.currY-1 == program.limits[2]:
                    if 'Down' in program.allowed:
                        program.allowed.remove('Down')
                if program.currX+1 == program.limits[1]:
                    if 'Right' in program.allowed:
                        program.allowed.remove('Right')
                if program.currX-1 == program.limits[0]:
                    if 'Left' in program.allowed:
                        program.allowed.remove('Left')

                action = program.allowed[0]



        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.allowed = ['Left','Right','Down','Up']
    program.currX = 0
    program.currY = 0
    program.cleanedRows = []
    program.cleanedColls = []
    program.limits = [99,99,99,99] #Left,Right,Down,Up

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt