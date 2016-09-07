import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."

    oldPercepts = [('None', 'Clean', 'Right', 'Left', 'atTop')]

    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        wall, status = percept
        lastLoc = oldPercepts[-1]

        lastAction = oldActions[-1]

        # checks to see if the starting point is dirty
        # if so cleans it

        if status == 'Dirty':
            action = 'Suck'
        else:
            # actions for the vacuum if at top
            if status == 'atTop':
               # checks to see if it hits the top
                if status == 'Bump':
                    action = 'Left'

                else:
                    action = 'Up'
                    # moves for the vacuum at the latest location of right
                    # tracks the moves starting with right if not at all

            elif lastLoc == 'Right':
                lastAction = oldActions[-2]
                if lastAction == 'Left':
                    action = 'Right'
                elif wall == 'Bump':
                    action = 'Left'
                    action = 'Down'
                else:
                    action = 'Right'

                    # moves for the vacuum at the latest location of left
                    # tracks the moves starting with left if not at wall
            elif lastLoc == 'Left':

                lastAction = oldActions[-2]
                if lastAction == 'Up' or lastAction == 'Down' :
                    action = 'Left'
                elif wall == 'Bump':
                    lastLoc = 'Right'
                    action = 'Right'

                else:
                    action = 'Left'
            else:
                action = 'Right'


        oldPercepts.append([wall, status, lastLoc]) # add all the old percepts that the agent has ever recieved
        oldActions.append(action) # add all the old actions of the agent

        return action
    return ag.Agent(program)