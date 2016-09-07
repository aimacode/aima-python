import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus = program.oldPercepts[-1]
            lastBump2, lastStatus2 = program.oldPercepts[-2]
            lastAction = program.oldActions[-1]
            lastAction2 = program.oldActions[-2]
            # Useless: if bump == 'Bump' and lastBump == 'Bump' and lastBump2 == 'Bump':
               # action = 'Up'
            #else: Useless
            #Works:
            # if bump == 'Bump' and lastBump == 'Bump' and (lastAction == 'Right' or lastAction == 'Left'):
            #     action = 'Down'
            # else:
            #     if bump == 'None' and lastAction != 'Suck': #and lastBump == 'None'
            #         action = lastAction
            #     else:
            #         if bump == 'None' and lastAction != 'Suck': # and lastBump == 'Bump'
            #             action = switchAction(lastAction)
            #         else:
            #             if bump == 'Bump' and lastAction != 'Suck':
            #                 action = switchAction(lastAction)
            #             else:
            #                 action = lastAction2
            if program.left == 'false' and bump == 'None':
                action = 'Left'
            else:
                if program.left == 'false' and bump == 'Bump':
                    program.left = 'true'
                    action = 'Right'
                else:
                    if program.right == 'false' and bump == 'None':
                        action = 'Right'
                    else:
                        if program.right == 'false' and bump == 'Bump':
                            program.right = 'true'
                            action = "Down"
                        else:
                            if program.down == 'false' and bump == 'None':
                                action = 'Down'
                            else:
                                if program.down == 'false' and bump == 'Bump':
                                    program.down = 'true'
                                    action = 'Up'
                                else:
                                    if program.up == 'false' and bump == 'None':
                                        action = 'Up'
                                    else:
                                        if program.up == 'false' and bump == 'Bump':
                                            program.up = 'true'
                                            action = "Down"
                                        else:
                                            if bump == 'None' and lastAction != 'Suck' and lastBump == 'None':
                                                action = lastAction
                                            else:
                                                if bump == 'None' and lastAction != 'Suck' and lastBump == 'Bump' and lastAction == 'Down' and program.lastDown == 'Left':
                                                    lastAction = 'Down2'
                                                    action = switchAction(lastAction)
                                                    program.lastDown = action
                                                else:
                                                    if bump == 'None' and lastAction != 'Suck' and lastBump == 'Bump' and lastAction == 'Down':
                                                        action = switchAction(lastAction)
                                                        program.lastDown = action
                                                    else:
                                                        if bump == 'Bump' and lastAction == 'Down' and program.lastDown =='Left':
                                                            lastAction = 'Down2'
                                                            action = switchAction(lastAction)
                                                            program.lastDown = action
                                                        else:
                                                            if bump == 'Bump' and lastAction == 'Down':
                                                                action = switchAction(lastAction)
                                                                program.lastDown = action
                                                            else:
                                                                if bump == 'Bump' and lastAction != 'Suck':
                                                                    action = switchAction(lastAction)
                                                                else:
                                                                    action = lastAction2






        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean'),('None', 'Clean')]
    program.oldActions = ['Right', 'Right']
    program.right = 'false'
    program.left = 'false'
    program.up = 'false'
    program.down = 'false'
    program.lastDown = ''

    # def switchAction(action):
    #     if action == 'Right':
    #         newAction = 'Left'
    #     if action == 'Left':
    #         newAction = 'Right'
    #     if action == 'Down':
    #         newAction = 'Up'
    #     if action == 'Up':
    #         newAction = 'Down'
    #     return newAction
    def switchAction(action):
        if action == 'Down'
            newAction = 'Left'
        if action == 'Left'
            newAction = 'Down'
        if action == 'Right'
            newAction = 'Down'
        if action == 'Down2'
            newAction = 'Right'
    switchAction.newAction = ''

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt