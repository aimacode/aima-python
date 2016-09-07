import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastBump2, lastStatus2 = program.oldPercepts[-2]
            if bump == 'Bump' and lastBump == 'Bump' and lastBump2 == 'Bump':
                action = 'Up'
            else:
                if bump == 'Bump' and lastBump == 'Bump':
                    action = 'Down'
                else:
                    if bump == 'None':
                        action = 'Right'
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