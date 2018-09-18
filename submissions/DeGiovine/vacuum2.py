import agents as ag

def HW2Agent() -> object:

    def program(percept):

        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]


            lastBump, lastStatus,  = program.oldPercepts[-1]
            if bump == 'None':
                action = 'Right'
            if bump == 'None' and lastAction == 'Up':
                action = 'Down'
            if bump == 'None' and lastAction == 'Left':
                action = 'Right'

            elif bump == 'Bump' and lastAction == 'Right':
                action = 'Down'
            elif bump == 'Bump' and lastAction == 'Left':
                action = 'Up'
            elif bump == 'Bump' and lastAction == 'Up':
                action = 'Right'
            elif bump == 'Bump' and lastAction == 'Down':
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
