import agents as ag

def HW2Agent() -> object:

    def program(percept):
        bump, status = percept
        lastBump, lastStatus = program.oldPercepts[-1]

        if status == 'Dirty':
            action = 'Suck'
        elif status == 'Clean' and lastBump == 'None':
            action = 'Right'
        elif status == 'Clean' and lastBump == 'Bump' :
            action = 'Left'
        elif lastStatus == 'Clean' and lastBump == 'None' :
            action ='Left'

        # else:
        #     if lastBump == 'None':
        #         action = 'Right'

                # checkHeight = True
                # if lastBump == 'Bump' and checkHeight == True :
                #     action = 'Down'

                # checkWidth = True
                # if lastBump == 'Bump'  and checkWidth == True :
                #     action = 'Left'
                # elif lastBump == 'None' and checkWidth == True :
                #     action = 'Left'


        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    # program.width = [0]
    # program.checkHeight = [False]
    # program.checkWidth = [False]


    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt