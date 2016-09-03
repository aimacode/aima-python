import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']
    score = 0
    #while (score < (90 and (date <= datetime.strptime('Sep 7 2016  6:00AM', '%b %d %Y %I:%M%p')))):


        def program(percept):
            "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
            bump, status = percept
            if status == 'Dirty':
                action = 'Suck'
            else:
                lastBump, lastStatus = oldPercepts[-1]
                if lastBump == 'None':
                     action = 'Right'
                else:
                    action = 'Left'
            oldPercepts.append(percept)
            oldActions.append(action)
            return action
        return ag.Agent(program)