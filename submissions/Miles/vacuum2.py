import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean', 'Right', 'Left', 'Up', 'Down')]
    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        wall, status = percept
        lastLoc = oldPercepts[-1]
        lastAction = oldActions[-1]
        # checks to see if the starting point is dirty
        if status == 'Dirty':
            action = 'Suck'
        else:

            if lastLoc == 'Up':
                if wall == 'Bump':  # checks to see if it hits the top
                    action = 'Down'
                    action = 'Left'
 # lastAction = 'Bump'
                else:
                    action = 'Up'
                    # moves for the vacuum at the latest location of right
            elif lastLoc == 'Right':


                lastAction = oldActions[-1]
                if lastAction == 'Left':
                    action = 'Right'
                elif wall == 'Bump':
                    lastLoc = 'Left'
                    action = 'Down'
                else:
                    action = 'Right'

                    # moves for the vacuum at the latest location of left
            elif lastLoc == 'Left' :

                lastAction = oldActions[-1]
                if lastAction == 'Up' or lastAction == 'Down' :
                    action = 'Left'
                elif wall == 'Bump' :
                    lastLoc = 'Right'
                    action = 'Right'

                else:
                    action = 'Left'
            else:
                action = 'Right'


        oldPercepts.append([wall, status, lastLoc])
        oldActions.append(action)

        return oldActions
    return ag.Agent(program)