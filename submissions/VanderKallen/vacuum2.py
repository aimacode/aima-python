import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']
    oldRight = ['None']
    oldLeft = ['None']
    oldUp = ['None']
    oldDown = ['None']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = oldPercepts[-1]
            if lastBump == 'Bump':
                if lastStatus == 'Right':
                    oldRight = 'Bump'
                if lastStatus == 'Left':
                    oldLeft = 'Bump'
            if oldRight =='Bump' & oldLeft == 'Bump':
                action = 'Down'
                oldLeft = 'None'
            if oldRight == 'None':
                action = 'Right'
            if oldLeft == 'None':
                action = 'Left'
            if oldLeft == 'Bump':
                action = 'Down'
                oldRight = 'None'

        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)