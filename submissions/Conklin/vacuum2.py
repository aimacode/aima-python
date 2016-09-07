import agents as ag
def HW2Agent() -> object:
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']

    def program(percept):
        wall, status = percept
        myAction = oldActions[-1]
        if myAction == 'NoOp':
            program.do = 'Left'
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = oldPercepts[-1]
            if wall != 'Bump':
                    #Effectively, if I'm at the start, go to the Right if possible....
                    if myAction == 'Suck':
                        if oldActions[-2] == 'NoOp':
                            action = 'Right'
                        else:
                            action = oldActions[-2]
                    #...if not, go to the Left
                    else:
                        if oldActions == 'NoOp':
                            action = 'Left'
                        else:
                            action = program.do

            elif wall == 'Bump':
                #If I hit the wall coming from the right, go to the Left
                if myAction == 'Right':
                    program.do = 'Left'
                    action = 'Left'
                #If I hit the wall coming from the left, go to the Right
                elif myAction == 'Left':
                    program.do = 'Right'
                    action = 'Right'
                #If I hit the wall coming Up, go Down
                elif myAction == 'Up':
                    program.do = 'Down'
                    action = 'Down'
                #If I hit the wall coming Down, go Up
                elif myAction == 'Down':
                    program.do = 'Up'
                    action = 'Up'


        oldPercepts.append(percept)
        oldActions.append(action)
        return action
    return ag.Agent(program)