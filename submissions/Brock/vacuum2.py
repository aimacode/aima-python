import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]
            if bump == 'None':
                if program.movingV == 'Down' and program.length != 0:
                    action = 'Down'
                    program.length -= 1
                else:
                    if program.movingH == 'Left':
                        if program.bumpleft == False:
                            action = 'Left'
                        else:
                            if program.widthTemp == 0:
                                program.widthTemp = program.width
                                program.movingH = 'Right'
                                if program.movingV == 'Up':
                                    action = 'Up'
                                    program.length += 1
                                else:
                                     action = 'Down'
                            else:
                                program.widthTemp -= 1
                                action = 'Left'

                    else:
                        if program.bumpright == False:
                            action = 'Right'
                            program.width += 1
                        else:
                            if program.widthTemp == 0:
                                program.widthTemp = program.width
                                program.movingH = 'Left'
                                if program.movingV == 'Up':
                                    action = 'Up'
                                    program.length += 1
                                else:
                                     action = 'Down'
                            else:
                                program.widthTemp -= 1
                                action = 'Right'
            else:
                if program.bumpleft == False:
                    if program.oldActions[-1] == 'Left':
                        program.bumpleft = True
                        program.movingH = 'Right'
                        action = 'Right'
                        #program.width += 1
                else:
                    if program.oldActions[-1] == 'Left':
                         if program.movingV == 'Up':
                             program.length += 1
                         action = program.movingV
                if program.bumpright == False:
                    if program.oldActions[-1] == 'Right':
                        program.widthTemp = program.width
                        program.bumpright = True
                        program.movingH = 'Left'
                        action = 'Up'
                        program.length += 1
                else:
                    if program.oldActions[-1] == 'Right':
                        if program.movingV == 'Up':
                            program.length += 1
                        action = program.movingV
                if program.bumpup == False:
                    if program.oldActions[-1] == 'Up':
                        program.bumpup = True
                        program.movingV = 'Down'
                        action = 'Down'
                        program.length -= 1
                else:
                    if program.oldActions[-1] == 'Down':
                        program.movingV = 'Down'
                        action = 'Down'
                        program.length -= 1
                if program.bumpdown == False:
                    if program.oldActions[-1] == 'Down':
                        program.bumpdown = True
                        program.movingV = 'Down'
                        action = program.movingH

        print(action)
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.counter = 0
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.location = 'None'
    program.movingH = 'Left'
    program.movingV = 'Up'
    program.bumpleft = False
    program.bumpright = False
    program.bumpup = False
    program.bumpdown = False
    #program.useLnumber = False
    #program.useRnumber = False
    #program.useUnumber = False
    #program.useDnumber = False
    program.length = 0
    #program.lengthTemp = 100
    program.width = 0
    program.widthTemp = 100


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt
