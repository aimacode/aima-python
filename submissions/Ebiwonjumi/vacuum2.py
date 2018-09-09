import agents as ag

def HW2Agent() -> object:

    def program(percept):
        try:
            lastlastAction = program.oldActions[-2]
        except IndexError:
            lastlastAction = False

        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if bump == 'None':

                action = 'Right'
            else:
                action = 'Left'

            if lastAction == 'Right' and bump == 'None':
                action = 'Right'

            if lastAction == 'Left' and bump == 'None':
                action = 'Left'

            if lastAction == 'Left' and bump == 'Bump':
                action = 'Down'

            if lastAction == 'Right' and bump == 'Bump':
                action = 'Up'

            if lastAction == 'Up' and bump == 'Bump':
                action = 'Left'

            if lastAction == 'Down' and bump == 'Bump':
                action = 'Right'

            if lastAction == 'Suck' and lastlastAction == 'Right':
                action = 'Right'

            elif lastAction == 'Suck' and lastlastAction == 'Left':
                action = 'Left'

            elif lastAction == 'Suck':
                action = 'Right'

            if lastAction == 'Suck':
                action = 'Left'




        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        program.counter += 1
        return action

    # assign static variables here
    program.counter = 0
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt