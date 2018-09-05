import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            lastAction2 = program.oldActions[-2]
            if bump == 'None': #and program.nextLevel != True:
                action = 'Left'
                if lastAction == 'Left':
                    action = 'Left'
                if lastAction == 'Right':
                    action = "Right"
                if lastAction == 'Up':
                    action = "Up"
                if lastAction == 'Down':
                    action = "Down"
            if bump != 'None': #and program.nextLevel != True:
                action = 'Up'
                if lastAction == 'Up':
                    action = 'Right'
                if lastAction == 'Right':
                    action = 'Down'
                if lastAction == 'Down':
                    action = 'Left'
                    #program.nextLevel = True
                #if program.nextLevel and lastAction == 'Left':
                    #action = 'Up'
            #if bump == 'None': #and program.nextLevel:
            #    if lastAction == 'Up':
            #        action = 'Right'
            #    if lastAction == 'Right':
            #        action = 'Right'

            if lastAction == 'Suck':
                action = lastAction2




        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        # program.oldDirection.append(direction)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp', 'Left', 'Left']
    program.nextLevel = False
    program.goingRight = False



    agt = ag.Agent(program)
    # assign class attributes here:
    agt.direction = ag.Direction('left')

    return agt