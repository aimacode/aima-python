import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:

            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            '''
            # score: 90.5
            if bump == 'None' and lastAction == 'NoOp':
                action = program.direction

            elif bump == 'Bump' and program.direction == 'Right':
                program.direction = 'Down'
                action = 'Down'
            elif bump == 'Bump' and program.direction == 'Down':
                program.direction = 'Left'
                action = 'Left'
            elif bump == 'Bump' and program.direction == 'Left':
                program.direction = 'Up'
                action = 'Up'
            elif bump == 'Bump' and program.direction == 'Up':
                program.direction = 'Right'
                action = 'Right'

            else:
                action = program.direction
            '''

            if program.bottomLeftCorner is False:
                if bump == 'None' and lastAction == 'NoOp':
                    action = program.direction
                elif bump == 'Bump' and program.direction == 'Right':
                    program.direction = 'Down'
                    action = 'Down'
                elif bump == 'Bump' and program.direction == 'Down':
                    program.direction = 'Left'
                    action = 'Left'
                elif bump == 'Bump' and program.direction == 'Left':
                    program.direction = 'Up'
                    action = 'Up'
                    program.bottomLeftCorner = True
                else:
                    action = program.direction
            else:
                if bump == 'None' and program.direction == 'Up':
                    program.direction = 'Right'
                    action = 'Right'
                elif bump == 'Bump' and program.direction == 'Left':
                    program.direction = 'Right'
                    action = 'Up'
                elif bump == 'None' and program.direction == 'Right' and lastAction == 'Up':
                    action = 'Right'
                elif bump == 'Bump' and program.direction == 'Right':
                    program.direction = 'Left'
                    action = 'Up'
                elif bump == 'None' and program.direction == 'Left' and lastAction == 'Up':
                    action = 'Left'

                else:
                    action = program.direction

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action       # must do this

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.direction = 'Right'
    program.bottomLeftCorner = False

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt