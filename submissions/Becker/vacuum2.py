import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        lastAction = program.oldActions[-1]
        lastBump, lastStatus = program.oldPercepts[-1]

        if status == 'Dirty':
            action = 'Suck'
        else:
            if not program.top:
                print('Going Up')
                if bump == 'Bump':
                    program.top = True
                    action = 'Left'
                else:
                    action = 'Up'
            elif program.direction == 'Left':
                print('Going Left')
                if bump == 'Bump':
                    program.direction = 'Right'
                    action = 'Right'
                else:
                    action = 'Left'
            elif program.direction == 'Right':
                print('Going Right')
                if bump == 'Bump':
                    program.direction = 'Left'
                    action = 'Down'
                else:
                    action = 'Right'


        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.top = False
    program.direction = 'Left'

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt