import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if lastAction == 'Suck' :
                action = program.oldActions[-2]
            elif (lastAction == 'Right' and bump == 'None'):
                action = 'Right'
            elif (lastAction == 'Right' and bump == 'Bump'):
                action = 'Left'
            elif (lastAction == 'Left' and bump == 'None') :
                action ='Left'
            elif (lastAction == 'Left' and bump == 'Bump') :
                action = 'Right'
            else:
                action = 'Left'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['Left']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt