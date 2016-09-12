import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        directions = ['Up', 'Left', 'Down', 'Right']
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = oldPercepts[-1]
            olderBump, olderStatus = oldPercepts[-2]

            if lastBump == 'None' and bump == 'none':
                action = directions[0]
            else:
                action = directions[2]
            if lastBump == 'Bump' and bump == 'Bump':
                action = directions[1]
            if lastBump == 'Bump' and bump == 'Bump' and olderBump == 'Bump':
                action = directions[3]


        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)