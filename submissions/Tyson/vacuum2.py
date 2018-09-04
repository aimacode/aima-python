import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status, = percept

        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus, = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if bump == 'None' and lastAction != 'Left':
                action = 'Right'

            else:

                if bump != 'None':
                    action = 'Left'

                else:
                    if lastAction == 'Left':
                        action = 'Left'
                    else:
                        action = 'Down'





        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean',)]
    program.oldActions = ['NoOp']


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt