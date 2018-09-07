import agents as ag
def HW2Agent() -> object:
    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            vertical, horiz = program.direction
            if lastAction == 'NoOp':
                action = vertical
            elif lastStatus == 'Dirty':
                action = horiz
            elif bump == 'None':
                if lastAction == 'Up':
                    action = 'Up'
                    program.direction = ('Up', horiz)
                elif lastAction == 'Down':
                    action = 'Left' if horiz == 'Right' else 'Right'
                    program.direction = (vertical, action)
                else:
                    action = lastAction
                    program.direction = (vertical, horiz)
            else:
                if lastBump:
                    #we hit a wall both vertically and horizontally
                    if lastAction == 'Down':
                        action = 'Up'
                        program.direction = (action, horiz)
                    elif lastAction == 'Up':
                        action = 'Left' if horiz == 'Right' else 'Right'
                        program.direction = ('Down', action)
                    else:
                        action = vertical
                        program.direction = (action, horiz)
                else:
                    action = vertical
                    program.direction = (action, horiz)
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.direction = ('Up', 'Right')

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')
    return agt
