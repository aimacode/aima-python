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
            if lastAction == 'Left' and bump != 'None':
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
            elif lastAction == 'Up':
                action = 'Right'
            elif lastAction == 'Left':
                action = 'Left'
            elif lastAction == 'Down' or bump != 'None':
                action = 'Down'
            elif lastBump != 'None' and lastStatus == 'Clean' and lastAction == 'Down':
                action = 'Left'
            elif lastAction == 'Up' or bump != 'None':
                action = 'Right'
            elif lastAction == 'Left':
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