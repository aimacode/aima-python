import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status, = percept
        hitTop = program.Counter
        rightBump = program.Counter2
        leftBump = program.Counter3
        lastBump, lastStatus, = program.oldPercepts[-1]
        lastAction = program.oldActions[-1]

        if status == 'Dirty':
            action = 'Suck'
        else:
            if rightBump ==2 and bump =='None':
                action = 'Right'
            else:
                if rightBump == 2 and lastAction == 'Right' and bump != 'None':
                    program.Counter2 = 3
                    program.Counter3 = 2
                    action = 'Down'
                else:
                    if leftBump == 2 and bump == "None":
                        action = 'Left'
                    else:
                        if leftBump == 2 and bump != 'None':
                            program.Counter2 = 2
                            action = 'Down'
                        else:
                            if bump == 'None' and hitTop ==0:
                                action = 'Up'
                            else:
                                    program.Counter = 1


                                    if lastAction == 'Up' and bump != 'None':
                                        action = 'Right'

                                    else:
                                        if rightBump == 0 and bump =='None':
                                            action = 'Right'

                                        else:
                                            if bump != 'None' and lastAction != 'Left':
                                                program.Counter2 = 1
                                                action = 'Left'
                                            else:
                                                if rightBump == 1 and bump == 'None':
                                                    action = 'Left'
                                                else:
                                                    if bump != 'None' and lastAction == 'Left':
                                                        program.Counter3 = 2
                                                        program.Counter2 = 2
                                                        action = 'Down'



















                        # if bump != 'None' and lastAction != 'Left':
                        #     program.Counter2 = 1
                        #     action = 'Left'
                        #
                        # else:
                        #     if lastStatus == 'Dirty' and rightBump ==0:
                        #         action = 'Right'
                        #     else:
                        #         if bump == 'None' and lastAction == 'Left' or lastAction == 'Suck':
                        #             action = 'Left'
                        #         else:
                        #             program.Counter2 = 0
                        #             action = 'Down'





        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean',)]
    program.oldActions = ['NoOp']
    program.Counter = 0
    program.Counter2 = 0
    program.Counter3 = 0


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt