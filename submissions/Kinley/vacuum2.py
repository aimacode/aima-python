import agents as ag

def HW2Agent() -> object:

    def program(percept):

        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastAction = program.oldActions[-1]
            if lastAction == 'Suck':
                action = program.oldActions[-2]

            elif bump == 'Bump' and lastAction == 'Left':
                action = 'Right'

            elif bump == 'Bump' and lastAction == 'Right':
                action = 'Down'

            elif bump == 'Bump' and lastAction == 'Down':
                action = 'Up'

            elif bump == 'Bump' and lastAction == 'Up':
                action = 'Left'

            elif bump == 'None' and lastAction == 'Left':
                action = 'Right'

            elif bump == 'None' and lastAction == 'Left':
                action = 'Left'

            elif bump == 'None' and lastAction == 'Right':
                action = 'Right'

            elif bump == 'None' and lastAction == 'Up':
                action = 'Up'

            elif bump == 'None' and lastAction == 'Down':
                action = 'Down'
                
            else:
                action = 'Right'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

# assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['Up']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt