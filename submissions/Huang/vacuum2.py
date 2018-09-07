import agents as ag

def HW2Agent() -> object:


    def program(percept):
        lastAction = program.oldActions[-1]
        lastBump, lastStatus, = program.oldPercepts[-1]
        bump, status = percept
        if lastAction == 'Up' and bump == 'None':
            program.rowCleaned = program.rowCleaned + 1
        if status == 'Dirty':
            action = 'Suck'
        elif program.rowCleaned > 0 and program.goDown == True:
            action = 'Down'
            program.rowCleaned =  program.rowCleaned - 1
        elif lastAction == 'Right' and bump == 'Bump' and program.goLeft == True:
            action = 'Left'
            program.goRight = False
        elif lastAction =='Left' and bump == 'None':
            action ='Left'
        elif lastAction == 'Left' and bump == 'Bump' and program.goDown == False:
            action = 'Up'
            program.goRight = True
            program.goLeft = False
        elif lastAction == 'Right' and bump == 'Bump' and program.goDown == False:
            action = 'Up'
            program.goRight = False
            program.goLeft = True
        elif lastAction == 'Left' and bump == 'Bump':
            action = 'Down'
            program.goRight = True
            program.goLeft = False
        elif lastAction == 'Right' and bump == 'Bump':
            action = 'Down'
            program.goRight = False
            program.goLeft = True
        elif bump == 'Bump':
            action ='Down'
            program.goDown = True
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            if bump == 'None' and program.goRight == True:
                action = 'Right'
            else:
                action = 'Left'


        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.rowCleaned = 0
    program.goDown = False
    program.goLeft = True
    program.goRight = True

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt