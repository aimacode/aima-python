import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean', 'topNotFound', 'moveLeft')]
    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        lastBump, lastStatus, topFound, moveDirection = oldPercepts[-1]
        lastAction = oldActions[-1]
        if status == 'Dirty':
            action = 'Suck'
        else:
            print (oldPercepts[-1])
            if topFound == 'topNotFound' :
                if bump == 'Bump' :
                    action = 'Left'
                    topFound = 'topFound'
                else:
                    action = 'Up'
            elif moveDirection == 'moveRight':
                print('inside the move right')
                lastAction2 = oldActions[-2]
                if lastAction2 == 'Left':
                    action = 'Right'
                elif bump == 'Bump':
                    moveDirection = 'moveLeft'
                    action = 'Down'
                else:
                    action = 'Right'
            elif moveDirection == 'moveLeft' :
                print('inside the move left')
                lastAction2 = oldActions[-2]
                if lastAction2 == 'Up' or lastAction == 'Down' :
                    action = 'Left'
                elif bump == 'Bump' :
                    moveDirection = 'moveRight'
                    action = 'Right'
                else:
                    action = 'Left'
            else:
                action = 'Right'
        oldPercepts.append([bump, status, topFound, moveDirection])
        oldActions.append(action)
        # print(lastBump, lastStatus, topFound, widthFound, moveDirection)
        return action
    return ag.Agent(program)