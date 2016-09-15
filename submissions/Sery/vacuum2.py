import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']

    actionScores = [{
        'Right': 0,
        'Left': 0,
        'Up': -1,
        'Down': -1,
        'NoOp': -100,
    }]
    level = 0

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        level = len(actionScores) - 1
        bump, status = percept
        lastBump, lastStatus = oldPercepts[-1]
        lastAction = oldActions[-1]

        if status == 'Dirty':
            action = 'Suck'
            actionScores[level][lastAction] += 2

        else:
            if bump == 'Bump':
                actionScores[level][lastAction] -= 10
            else:
                if lastAction == 'Up' or lastAction == 'Down':
                    actionScores.append({
                        'Right': 0,
                        'Left': 0,
                        'Up': -1,
                        'Down': -1,
                    })

            highest = -80
            for actionType, score in actionScores[level].items():
                if score > highest:
                    highest = score
                    action = actionType


        print(actionScores)

        oldPercepts.append(percept)
        oldActions.append(action)
        return action

    return ag.Agent(program)