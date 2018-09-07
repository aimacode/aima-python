import agents as ag
 def HW2Agent() -> object:
     def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            if program.count == 0:
                if bump == 'Bump':
                    program.count += 1
                    action = 'Down'
                else:
                    action = 'Right'
            elif program.count == 1:
                if bump == 'Bump':
                    program.count = 6
                    action = 'Left'
                else:
                    program.count += 1
                    action = 'Left'
            elif program.count == 2:
                if bump == 'Bump':
                    program.count += 1
                    action = 'Down'
                else:
                    action = 'Left'
            elif program.count == 3:
                if bump == 'Bump':
                    program.count = 7
                    action = 'Up'
                else:
                    program.count = 0
                    action = 'Right'
            # if program.count == 4:
            #     action = 'Up'
            #     if bump == 'Bump':
            #         program.count = 6
            #     else:
            # if program.count == 5:
            #     program.count = 7
            elif program.count == 6:
                if bump == 'Bump':
                    program.count += 1
                    action = 'Up'
                else:
                    action = 'Left'
            elif program.count == 7:
                if bump == 'Bump':
                    program.count = 3
                    action = 'Down'
                else:
                    program.count += 1
                    action = 'Right'
            elif program.count == 8:
                if bump == 'Bump':
                    program.count += 1
                    action = 'Up'
                else:
                    action = 'Right'
            elif program.count == 9:
                if bump == 'Bump':
                    program.count = 1
                    action = 'Down'
                else:
                    program.count = 6
                    action = 'Left'
             # if bump == 'None' and lastStatus == 'Clean':
            #     action = program.oldActions[-1]
            # elif bump == 'None' and lastStatus == 'Dirty':
            #     action = program.oldActions[-2]
            # else:
            #     program.lastWall.append(program.oldActions[-1])
            #     if program.lastWall[-1] == 'Right' and program.lastWall[-2] == 'Down':
            #         action = 'Up'
            #     elif program.lastWall[-1] == 'Left' and program.lastWall[-2] == 'Right':
            #         action = 'Up'
            #     elif program.lastWall[-1] == 'Up' and lastBump == 'Bump':
            #         action = 'Down'
            #     elif program.lastWall[-1] == 'Up' and lastBump == 'None':
            #         action = 'Right'
            #     elif program.lastWall[-1] == 'Right' and program.lastWall[-2] == 'Up':
            #         action = 'Left'
            #     elif program.lastWall[-1] == 'Right' and program.lastWall[-2] == 'Left':
            #         action = 'Up'
            #     elif program.lastWall[-1] == 'Left' and program.lastWall[-2] == 'Down':
            #         action = 'Right'
            #     elif program.lastWall[-1] == 'Down' and program.lastWall[-2] == 'None':
            #         action = 'Right'
            #
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action
     # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['Left', 'Right']
    program.count = 0
    # program.lastWall = ['None', 'Down']
     agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')
     return agt
