import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            if program.count == 0:
                if bump == 'Bump':
                    action = 'Down'
                    program.count += 1
                else:
                    action = 'Right'
            elif program.count == 1:
                if bump == 'Bump':
                    action = 'Right'
                    program.count += 1
                else:
                    action = 'Left'
                    program.count = 4
            elif program.count == 2:
                if bump == 'Bump':
                    action = 'Up'
                    program.count += 1
                else:
                    action = 'Left'
            elif program.count == 3:
                if bump == 'Bump':
                    action = 'Up'
                    program.count = 5
                else:
                    action = 'Right'
                    program.count = 0
            elif program.count == 4:
                if bump == 'Bump':
                    action = ''
                    program.count += 1
                if bump == 'None':
                    action = 'Left'
            elif program.count == 5:
                if bump == 'Bump':
                    action = 'Down'
                    program.count = 3
                if bump == 'None':
                    action = 'Right'
                    program.count += 1
            elif program.count == 6:
                if bump == 'Bump':
                    action = 'Up'
                    program.count += 1
                if bump == 'None':
                    action = 'Right'
            elif program.count == 7:
                if bump == 'Bump':
                    program.count = 1
                    action = 'Down'
                if bump == 'None':
                    action = 'Left'
                    program.count = 4

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.count = 0

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt