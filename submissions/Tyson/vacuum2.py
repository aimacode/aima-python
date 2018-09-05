import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status, = percept
        hitTop = program.Counter

        if status == 'Dirty':
            action = 'Suck'
        else:
            if bump == 'None' and hitTop ==0:
                action = 'Up'
            else:
                program.Counter = 1
                lastBump, lastStatus, = program.oldPercepts[-1]
                lastAction = program.oldActions[-1]

                if lastAction == 'Up' or lastAction == 'Down' or lastAction == 'Right' and bump == 'None' :
                    action = 'Right'

                else:

                    if bump != 'None' and lastAction != 'Left':
                        action = 'Left'

                    else:
                        if bump == 'None' and lastAction == 'Left' or lastAction == 'Suck':
                            action = 'Left'
                        else:
                            action = 'Down'





        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean',)]
    program.oldActions = ['NoOp']
    program.Counter = 0


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt