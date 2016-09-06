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
            lastAction = oldActions[-1]
            secondToLastAction = oldActions[len(oldActions)-2]
            lastBump, lastStatus = oldPercepts[-1]
            if lastAction == 'Left' and bump == "Bump":
                action = 'Up'
            else:
                action = 'Left'
            if lastAction == 'Up':
                action = 'Right'
            if secondToLastAction == 'Up':
                action = 'Down'
            # print(secondToLastAction)


        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)