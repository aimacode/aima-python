import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if len(program.allowed)==0:
                program.allowed = ['Left', 'Right', 'Down', 'Up']
            if bump == 'None':
                action = program.allowed[0]
            else:
                program.allowed.remove(lastAction)
                if len(program.allowed) == 0:
                    program.allowed = ['Left', 'Right', 'Down', 'Up']
                    program.allowed.remove(lastAction)
                action = program.allowed[0]

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.allowed = ['Left','Right','Down','Up']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt