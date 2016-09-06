import agents as ag


def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    #the first action is a lie!!!
    oldActions = ['Left']
    #while (score < (90 and (date <= datetime.strptime('Sep 7 2016  6:00AM', '%b %d %Y %I:%M%p')))):

    def program(percept):
            "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
            bump, status = percept
            if status == 'Dirty':
                action = 'Suck'
            elif bump == 'None':
                if oldActions[(oldActions.__len__() - 1)] == 'Suck':
                    if oldActions[(oldActions.__len__() - 2)] != 'Up':
                        action = oldActions[(oldActions.__len__() - 2)]
                    else:
                        if oldActions[(oldActions.__len__() - 3)] == 'Right':
                            action  = 'Left'
                        else: action = 'Right'

                elif oldActions[(oldActions.__len__() - 1)] == 'Up' and oldActions[(oldActions.__len__() - 2)] == 'Right':
                    action = 'Left'
                elif oldActions[(oldActions.__len__() - 1)] == 'Up' and oldActions[(oldActions.__len__() - 2)] == 'Left':
                    action = 'Right'
                else:
                    action = oldActions[(oldActions.__len__() - 1)]
            elif bump == 'Bump':
                if oldActions[(oldActions.__len__() - 1)] == 'Right':
                    action = 'Up'
                if oldActions[(oldActions.__len__() - 1)] == 'Up' and oldActions[(oldActions.__len__() - 2)] == 'Left':
                    action = 'Right'
                elif oldActions[(oldActions.__len__() - 1)] == 'Up' and oldActions[(oldActions.__len__() - 2)] == 'Right':
                    action = 'Left'
                elif oldActions[(oldActions.__len__() - 1)] == 'Up':
                    action = 'Right'
                if oldActions[(oldActions.__len__() - 1)] == 'Down':
                    action = 'Right'
                if oldActions[(oldActions.__len__() - 1)] == 'Left':
                    action = 'Down'
                    for x in range (0, oldActions.__len__()):
                        if oldActions[x] == 'Down':
                            action = 'Up'
                            break

            oldPercepts.append(percept)
            oldActions.append(action)
            return action
    return ag.Agent(program)
