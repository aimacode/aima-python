import agents as ag
#Braden Carei
#This file with move through the virtual space by
#moving down in a back and forth pattern til it hits the bottom
#then return to original line and go up in same motion.
def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            program.vertCount -= 1
            action = 'Suck'
        else:
            program.vertCount-=1
            lastBump, lastStatus,  = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]

            if lastAction == 'NoOp':
                action = program.horizontal
            elif lastStatus == 'Dirty' and program.oldActions[-2]==program.vertical:
                if program.vertCount ==1:
                    if program.horizontal == 'Right':
                        program.horizontal = 'Left'
                    else:
                        program.horizontal = 'Right'
                action = program.horizontal
            elif lastStatus == 'Dirty' and program.oldActions[-2] == program.horizontal:
                action = program.horizontal
            elif lastStatus == 'Dirty':
                action = program.horizontal
            elif bump == 'None':
                if lastAction == 'Down':
                    program.downCount+=1
                    if program.horizontal == 'Right':
                        program.horizontal = 'Left'
                    else:
                        program.horizontal = 'Right'
                    action = program.horizontal

                elif lastAction == 'Up' and program.downCount == 0:
                    if program.horizontal == 'Right':
                        program.horizontal = 'Left'
                    else:
                        program.horizontal = 'Right'
                    action = program.horizontal

                elif lastAction == 'Up' and program.downCount>0:

                    action = program.vertical
                    program.downCount -= 1
                    program.vertCount = 3

                else:
                    action = lastAction
            else:
                if lastBump:
                    # we've bumped multiple times, so we're in a corner
                    if lastAction == 'Down' and program.horizontal == 'Right':
                        program.vertical = 'Up'
                        if program.horizontal == 'Right' :
                            program.horizontal = "Left"
                        else:
                            program.horizontal = "Right"

                        action = program.vertical
                        program.vertCount = 3
                    if lastAction == 'Down' and program.horizontal == 'Left':
                        program.vertical = 'Up'

                        action = program.vertical
                        program.vertCount = 3
                    elif lastAction == 'Up' and program.oldActions[-2]!= 'Down':
                        action = program.horizontal
                        if program.horizontal == 'Right':
                            program.horizontal = "Left"
                        else:
                            program.horizontal = "Right"
                    elif lastAction == 'Up' and program.oldActions[-2]== 'Down':
                        action = program.horizontal

                    else:
                        action = program.vertical
                        program.vertCount = 3;
                else:
                    if program.vertical == 'Down':
                        program.downCount+=1
                    if program.horizontal == 'Right':
                        program.horizontal = "Left"
                    else:
                        program.horizontal = "Right"
                    action = program.horizontal


        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.downCount = -1
    program.vertCount = 0
    program.horizontal = 'Right'
    program.vertical = 'Down'


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt