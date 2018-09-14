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
            if program.state == 0:
                if bump == 'Bump':
                    program.state = 1
                    action = 'Right'
                else:
                    action = 'Up'
            elif program.state == 1:
                if bump == 'Bump':
                    program.state = 2
                    action = 'Left'
                else:
                    action = 'Down'

            elif program.state == 2:
                if bump == 'Bump':
                    program.state = 3
                    action = 'Up'
                else:
                    action = 'Left'
            elif program.state == 3:
                if bump == 'Bump':
                    program.state = 2
                    action = 'Up'
                else:
                    action = 'Right'

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
