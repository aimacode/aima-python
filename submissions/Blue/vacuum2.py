import agents as ag

def HW2Agent() -> object:
    # program.oldPercepts = [('None', 'Clean','down', 0)]
    # program.oldActions = ['NoOp']
    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus, bottom, count = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            # program.bottom = 'down'
            # program.count = 0

            if program.bottom == 'down':
                if bump == 'None':
                    action = 'Down'
                elif bump == 'Bump' and lastAction[-1] == 'Down':
                    action = 'Left'
                    program.bottom = 'up'
                    action = 'Up'
                else:
                    program.bottom = 'right'
                    action = 'Right'
            if program.bottom == 'right':
                if bump =='None':
                   action = 'Right'
                else:
                    program.bottom = 'up'
                    action = 'Up'
            if program.bottom == 'up':
                if bump == 'None':
                    action = 'Up'
                elif bump == 'Bump' and lastAction[-1] == 'Up':
                    action = 'Left'
                    program.bottom = 'down'
                    action = 'Down'
                else:
                    program.bottom = 'left'
                    action = 'Left'

            # if program.bottom == 'left':
            # elif bump == 'Bump':
            #     action = 'Up'
            # else:
            #     action = 'Right'




        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean', 'down', 0)]
    program.oldActions = ['NoOp']
    # program.bottom = 'noBott'
    # program.count = 0

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt