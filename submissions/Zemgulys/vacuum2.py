import agents as ag


def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        lastBump, lastStatus, lastAction = program.oldPercepts[-1], program.oldPercepts[-1], program.oldActions[-1]

        if status == 'Dirty':
            if lastAction == 'Left':
                program.d = 0
                action = 'Suck'
            elif lastAction == 'Right':
                program.d = 1
                action = 'Suck'
            elif lastAction == 'Up':
                program.d = 2
                action = 'Suck'
            elif lastAction == 'Down':
                program.d = 3
                action = 'Suck'
            else:
                action = 'Suck'
        else:

            if bump == 'None':
                if lastAction == 'Left':
                    action = 'Left'
                elif lastAction == 'Down':
                    action = 'Down'
                elif lastAction == 'Up':
                    action = 'Up'
                else:
                    action = program.dir[program.d]

            else:
                if lastAction == 'Down':
                    action = 'Up'
                elif lastAction == 'Right':
                    action = 'Left'
                elif lastAction == 'Up':
                    action = 'Down'
                else:
                    action = 'Down'

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.dir = ['Left', 'Right', 'Up', 'Down']
    program.d = 1

    agt = ag.Agent(program)

    return agt
