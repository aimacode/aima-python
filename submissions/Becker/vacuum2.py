import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        # lastAction = program.oldActions[-1]
        # lastBump, lastStatus = program.oldPercepts[-1]

        if status == 'Dirty':
            action = 'Suck'
        else:
            if not program.top:
                print('Going Up')
                if bump == 'Bump':
                    program.top = True
                    action = 'Left'
                    program.step = program.step + 1
                else:
                    action = 'Up'
                    program.step = program.step + 1
            elif program.direction == 'Left':
                print('Going Left')
                if bump == 'Bump':
                    if program.oldDirection == 'Right':
                        program.direction = 'Right'
                        program.oldDirection = 'Left'
                        action = 'Down'
                        program.step = program.step + 1
                    elif program.oldDirection == 'Left':
                        program.direction = 'Right'
                        action = 'Right'
                        program.step = program.step + 1
                else:
                    action = 'Left'
                    program.step = program.step + 1
            elif program.direction == 'Right':
                print('Going Right')
                if bump == 'Bump':
                    if program.oldDirection == 'Left':
                        program.direction = 'Left'
                        program.oldDirection = 'Right'
                        action = 'Down'
                        program.step = program.step + 1
                    elif program.oldDirection == 'Right':
                        program.direction = 'Left'
                        action = 'Left'
                        program.step = program.step + 1
                else:
                    action = 'Right'
                    program.step = program.step + 1
            else:
                action = 'Right'
                program.step = program.step + 1


        print(program.step)
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.top = False
    program.direction = 'Left'
    program.oldDirection = 'Left'
    program.step = 1

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt