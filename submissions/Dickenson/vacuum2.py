import agents as ag

def HW2Agent() -> object:

    def program(percept):

        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            if lastAction == 'Suck':
                action = program.oldActions[-2]
            elif (bump == 'Bump' and (lastAction == 'Left' or lastAction == 'Right')):
                action = 'Down'
            elif ((lastAction == 'Down') and (bump == 'Bump')):
                action = 'Up'
            elif ((lastAction == 'Up') and (bump == 'Bump')):
                action = 'Right'
            elif ((lastAction == 'Right') and (bump != 'Bump')):
                action = 'Right'
            elif ((lastAction == 'Left') and (bump != 'Bump')):
                action = 'Left'
            elif ((lastAction == 'Up') and (bump != 'Bump')):
                action = 'Up'
            elif ((lastAction == 'Down') and (bump != 'Bump')):
                action = 'Down'
            else:
                action = 'Left'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

# assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt