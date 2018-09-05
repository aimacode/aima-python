import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
            direction = 'Right'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            # oldDirection = program.oldDirection[-1]
            if bump == 'None':
                action = 'Left'
                if lastAction == 'Left':
                    action = 'Left'
                if lastAction == 'Right':
                    action = "Right"
                if lastAction == 'Up':
                    action = "Up"
                if lastAction == 'Down':
                    action = "Down"
            if bump != 'None':
                action = 'Up'
                if lastAction == 'Up':
                    action = 'Up'
                if bump != 'None' and lastBump != 'None':
                    action = 'Right'
                    if lastAction == 'Right':
                        action = 'Down'
                        goingRight = True
                    #if lastAction == 'Right' and goingRight:
                        action = 'Down'




        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        # program.oldDirection.append(direction)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    goingRight = False


    agt = ag.Agent(program)
    # assign class attributes here:
    agt.direction = ag.Direction('left')

    return agt