import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            if bump == 'None':
                if program.bumpcount == 0:
                    action = 'Right'
                elif program.bumpcount == 1:
                    action = 'Down'
                elif program.bumpcount == 2:
                    action = 'Left'
                else:
                    action = 'Up'
            else:
                if bump == 'Bump' and program.oldActions[-1] == 'Right':
                    program.bumpcount += 1
                    action = 'Down'
                elif bump == 'Bump' and program.oldActions[-1] == 'Down':
                    program.bumpcount += 1
                    action = 'Left'
                elif bump == 'Bump' and program.oldActions[-1] == 'Left':
                    program.bumpcount += 1
                    action = 'Up'
                else:
                    program.bumpcount -= 3
                    action = 'Down'



        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.bumpcount = 0
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']

    agt = ag.Agent(program)
    # assign class attributes here:
    #agt.direction = ag.Direction('left')


    return agt