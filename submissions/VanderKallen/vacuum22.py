import agents as ag
def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        curAct = oldActions[-1]
        program.do = curAct
        if curAct == 'NoOp':
            program.do = 'Right'
        if status == 'Dirty':
            action = 'Suck'
        else:
            #lastBump, lastStatus = oldPercepts[-1]
            if bump == 'None':
                    if curAct == 'Suck':
                        if oldActions[-2] == 'NoOp':
                            action = 'Right'
                        else:
                            action = oldActions[-2]
                    else:
                        if oldActions == 'NoOp':
                            action = 'Right'
                        else:
                            action = program.do
            else:
                if bump == 'Bump':
                    if curAct == 'Right':
                        program.do = 'Left'
                        action = 'Left'
                    else:
                        if curAct == 'Down':
                            program.do = 'Up'
                            action = 'Up'
                        else:
                            if curAct == 'Up':
                                program.do = 'Right'
                                action = 'Right'
                            else:
                                if curAct == 'Left':
                                    program.do = 'Right'
                                    action = 'Down'

        oldPercepts.append(percept)
        oldActions.append(action)
        program.do = 'Right'
        return action
    return ag.Agent(program)