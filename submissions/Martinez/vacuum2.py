import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            if bump == 'None':
                action = 'Left'

            if bump != 'None':
                action = 'Right'

            if bump != 'None' and lastAction == 'Left':
                action = 'Right'

            if bump != 'None' and lastAction == 'Right':
                action = 'Down'

            if bump != 'None' and lastAction == 'Down':
                action = 'Up'

            if bump == 'None' and lastAction == 'Down':
                action = 'Down'

            if bump == 'None' and lastAction == 'Right':
                action = 'Right'

            if bump == 'None' and lastAction == 'Left':
                action = 'Right'

            if bump != 'None' and lastAction == 'Left':
                action = 'Up'
#it says local variable might be referenced before assingment?

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