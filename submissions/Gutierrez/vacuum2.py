import agents as ag

def HW2Agent() -> object:
    x = 0
    def program(percept):
        bump, status = percept
        lastAction = program.oldActions[-1]
        prevAction = program.oldActions[-2]
        # if lastAction == 'Suck':
        # lastAction = prevAction
        if status == 'Dirty':
            action = 'Suck'
        else:
            if lastAction == 'Suck':
                lastBump, lastStatus, = program.oldPercepts[-1]
                if prevAction == 'Right' or prevAction == 'NoOp' :
                    if bump == 'None':
                        action = 'Right'
                    else:
                        action = 'Up'
                if prevAction == 'Up':
                    if bump == 'None':
                        x = program.oldActions[-3]
                        action = x
                    else:
                        action = 'down'
                if prevAction == 'Left':
                    if bump == 'None':
                        action = 'Left'
                    else:
                        action = 'Up'
                if prevAction == 'Down':
                    if bump == 'None':
                        action = 'Down'
                    else:
                        action = 'Left'
            else:
                lastBump, lastStatus,  = program.oldPercepts[-1]
                if lastAction == 'NoOp' or lastAction == 'Right':
                    if bump == 'None':
                        action = 'Right'
                    else:
                        action = 'Up'

                if lastAction == 'Up':
                    if bump == 'None' and prevAction == 'Right':
                        action = 'Left'
                    else:
                        if bump == 'None' and prevAction == 'left':
                            action = 'Left'
                        else:
                            action = 'Down'
                if lastAction == 'Left':
                    if bump == 'None':
                        action = 'Left'
                    else:
                        action = 'Up'

                if lastAction == 'Down':
                    aAction = Downs.__sizeof__()
                    if bump == 'None':
                        action = 'Down'
                    else:
                        if aAction == 'left':
                            action = 'Right'
                        else:
                            action = 'Left'

        # print(prevAction)
        if action == 'Down':
            Downs.append(action)

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action


    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp', 'NoOp']
    Downs = [1]



    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt