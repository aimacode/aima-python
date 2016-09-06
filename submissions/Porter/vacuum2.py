import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = oldPercepts[-1]
            lastAction = oldActions [-1]
            # if lastBump == 'None':
            #     action = 'Right'
            # else:
            #     action = 'Left'

            if bump == 'Bump' and lastStatus == 'Clean':
                action = 'Left'
            else:
                action = 'Right'

        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)