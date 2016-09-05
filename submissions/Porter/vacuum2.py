import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']
    # oldLocation = [location]

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
                # elif status == 'Clean':
                #     action = 'Up'


            if lastAction == 'Left' and bump == 'Bump' and lastStatus == 'Clean':
                action = 'Right'
            else:
                action = 'Left'

        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)