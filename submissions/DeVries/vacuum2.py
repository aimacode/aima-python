import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            if (len(program.oldActions)>1):
                twoActionsAgo = program.oldActions[-2]
            if lastAction == 'Suck':
                lastAction = program.oldActions[-2]
                if (len(program.oldActions) > 2):
                    twoActionsAgo = program.oldActions[-3]
            if bump == 'None':
                if (lastAction == 'Left' or lastAction == 'Right'):
                    action = lastAction
                elif lastAction == 'Up':
                    if twoActionsAgo == 'Right':
                        action = 'Left'
                    else:
                        action = 'Right'
                elif lastAction == 'Down':
                    action = 'Down'
                else:
                    action = 'Right'
            elif lastAction == 'Up':
                if twoActionsAgo == 'Left':
                    program.directionAfterDownbump = 'Right'
                elif twoActionsAgo == 'Right':
                    program.directionAfterDownBump = 'Left'
                action = 'Down'
            elif lastAction == 'Down':
                action = program.directionAfterDownBump
            elif lastAction == 'Right' and twoActionsAgo == 'Down':
                action = 'Left'
            else:
                action = 'Up'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.directionAfterDownBump = 'Right'

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt
