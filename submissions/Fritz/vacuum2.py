import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."

    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]

            if bump == 'None':
                action = program.act
            else:
                if program.act == 'Up':
                    program.act = 'Right'
                    action = program.act
                else:
                    if program.act == 'Right':
                        program.act = 'Down'
                        action = program.act
                    else:
                        if program.act == 'Down':
                            program.act = 'Left'
                            action = program.act
                        else:
                            if program.act == 'Left':
                                program.act = 'Up'
                                action = program.act
               # program.top = program.top + 1




        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.counter = 0
    program.tall = 0
    program.wide = 0
    program.act = 'Up'

    return ag.Agent(program)