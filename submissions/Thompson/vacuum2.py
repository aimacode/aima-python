import agents as ag
#testsdjhfakdjfhds
def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            if program.counter == 0:
                if bump == 'Bump':
                    program.counter += 1
                    action = 'Right'
                else:
                    action = 'Down'
            elif program.counter == 1:
                if bump == 'Bump':
                    program.counter = 6
                    action = 'Up'
                else:
                    program.counter += 1
                    action = 'Up'
            elif program.counter == 2:
                if bump == 'Up':
                    program.counter += 1
                    action = 'Up'
                else:
                    action = 'Up'
            elif program.counter == 3:
                if bump == 'Bump':
                    program.counter = 7
                    action = 'Left'
                else:
                    program.counter = 0
                    action = 'Down'
            #Skipping 4 and 5 because it's similar to 1 and 3
            elif program.counter == 6:
                if bump == 'Bump':
                    program.counter += 1
                    action = 'Left'
                else:
                    action = 'Up'
            elif program.counter == 7:
                if bump == 'Bump':
                    program.counter = 3
                    action = 'Right'
                else:
                    program.counter += 1
                    action = 'Down'
            elif program.counter == 8:
                if bump == 'Bump':
                    program.counter += 1
                    action = 'Left'
                else:
                    action = 'Down'
            elif program.counter == 9:
                if bump == 'Bump':
                    program.counter = 1
                    action = 'Right'
                else:
                    program.counter = 6
                    action = 'Up'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['Left', 'Right']
    program.counter = 0
    # program.lastWall = ['None', 'Down']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt
