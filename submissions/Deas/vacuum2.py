import agents as ag
def HW2Agent() -> object:
    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            #lastBump, lastStatus,  = program.oldPercepts[-1]
            if program.state == 0 and bump == 'None':
                action = 'Right'
            elif program.state == 0 and bump == 'Bump':
                program.count = + 1
                action = 'Down'
                program.state = 1
            elif program.state == 1 and bump == 'None':
                #program.count = + 1
                action = 'Left'
                program.state = 2
            elif program.state == 1 and bump == 'Bump':
                program.count = - 1
                action = 'Up'
                program.state = 4
            elif program.state == 2 and bump == 'None':
                action = 'Left'
            elif program.state == 2 and bump == 'Bump':
                program.count = + 1
                action = 'Down'
                program.state = 3
            elif program.state == 3 and bump == 'None':
                #program.count = + 1
                action = 'Right'
                program.state = 0
            elif program.state == 3 and bump == 'Bump':
                program.count = - 1
                action = 'Up'
                program.state = 5
            elif program.state == 4 and program.count != 0:
                program.count = - 1
                action = 'Up'
            elif program.state == 4 and program.count == 0:
                action = 'Left'
                program.state = 6
            elif program.state == 5 and program.count != 0:
                program.count = - 1
                action = 'Up'
            elif program.state == 5 and program.count == 0:
                action = 'Right'
                program.state = 8
            elif program.state == 6 and bump == 'None':
                action = 'Left'
            elif program.state == 6 and bump == 'Bump':
                action = 'Up'
                program.state = 7
            elif program.state == 7 and bump == 'None':
                action = 'Right'
                program.state = 8
            elif program.state == 7 and bump == 'Bump':
                action = 'NoOp'
            elif program.state == 8 and bump == 'None':
                action = 'Right'
            elif program.state == 8 and bump == 'Bump':
                action = 'Up'
                program.state = 9
            elif program.state == 9 and bump == 'None':
                action = 'Left'
                program.state = 6
            elif program.state == 9 and bump == 'Bump':
                action = 'NoOp'
            else:
                action = 'NoOp'
         #program.oldPercepts.append(percept)
        #program.oldActions.append(action)
        return action
    # assign static variables here
    #program.oldPercepts = [('None', 'Clean')]
    #program.oldActions = ['NoOp']
    program.state = 0
    program.count = 0

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')
    return agt
