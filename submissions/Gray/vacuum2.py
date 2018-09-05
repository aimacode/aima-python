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
                    action = 'Right'
                if lastAction == 'Right':
                    action = 'Down'
                if lastAction == 'Down':
                    action = 'Left'
                    program.goingLeft = True
                    program.goingRight = False
            if bump == 'None' and (program.goingLeft or program.goingRight):
                if lastAction == 'Left':
                    action = 'Left'
                if lastAction == 'Right':
                    action = 'Right'
                if lastAction == 'Up' and program.goingLeft:
                    action = 'Left'
                if lastAction == 'Up' and program.goingRight:
                    action = 'Right'
            if bump != 'None' and (program.goingLeft or program.goingRight):
                if lastAction == 'Left':
                    action = 'Up'
                    program.goingRight = True
                    program.goingLeft = False
                if lastAction == 'Right':
                    action = 'Up'
                    program.goingRight = False
                    program.goingLeft = True
            if lastAction == 'Suck':
                action = lastAction2

        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp', 'Left', 'Left']
    program.nextLevel = False
    program.goingLeft = False
    program.goingRight = False



    agt = ag.Agent(program)
    # assign class attributes here:
    agt.direction = ag.Direction('left')

    return agt