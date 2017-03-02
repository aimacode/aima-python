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


            if lastAction != 'Suck':

                if lastAction == 'Left' and bump == 'Bump' and status == 'Clean':
                    action = 'Right'
                elif lastAction == 'Right' and bump == 'Bump' and status == 'Clean':
                    action = 'Up'
                elif lastAction == 'Up' and bump == 'Bump' and status == 'Clean':
                    action = 'Down'
                elif bump == 'None' and status == 'Clean' and lastStatus == 'Clean':
                    action = 'Right'
                else:
                    action = 'Left'

            else:
                if  bump == 'Bump' and status == 'Clean':
                    action = 'Right'
                elif bump == 'Bump' and status == 'Clean':
                    action = 'Up'
                elif bump == 'Bump' and status == 'Clean':
                    action = 'Down'
                elif bump == 'None' and status == 'Clean' and lastStatus == 'Clean':
                    action = 'Right'
                else:
                    action = 'Left'


            oldPercepts.append(percept)
            oldActions.append(action)
        return action
    return ag.Agent(program)