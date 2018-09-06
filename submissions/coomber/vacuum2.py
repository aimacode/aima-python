import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        lastBump, lastStatus, = program.oldPercepts[-1]

        if program.oldActions[-1]=='Right' and lastBump == 'Bump':
            program.rightBumped = True
        if program.oldActions[-1]=='Left' and lastBump == 'Bump':
            program.leftBumped = True
        if program.oldActions[-1]=='Down' and lastBump == 'Bump':
            program.downBumped = True
        if program.oldActions[-1]=='Up' and lastBump == 'Bump':
            program.upBumped = True

        if status == 'Dirty':
            action = 'Suck'
        else:
            if (bump == 'None') and (program.leftBumped == False):
                action = 'Left'
            elif (bump == 'Bump') and (program.oldActions[-1:] == 'Left'):
                program.leftBumped = True
            elif (bump == 'Bump') and (program.oldActions[-1:] == 'Right'):
                program.rightBumped = True
            elif (bump == 'None') and (program.rightBumped == False):
                action = 'Right'
            elif program.rightBumped and program.leftBumped:
                action = 'Down'
            elif (bump == 'Bump') and (program.oldActions[-1:]== 'Down'):
                program.downBumped = True;
            elif program.rightBumped and program.leftBumped and program.DownBumped:
                action = 'Up'
            else:
                action = 'Right'

            #(program.oldActions[-1] == 'Left') and (status == 'Bumped')

            #if bump == 'None':
            #    action = 'Left'
            #else:
            #    action = 'Right'



        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.leftBumped = False
    program.rightBumped = False
    program.upBumped = False
    program.downBumped = False

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt