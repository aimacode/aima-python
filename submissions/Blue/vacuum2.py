import agents as ag

def HW2Agent() -> object:
    oldPercepts = [('None', 'Clean','noTop')]
    oldActions = ['NoOp']
    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus, bottom = oldPercepts[-1]
            lastAction = oldActions[-1]

            if bump == 'None':
                action = 'Down'
            else:
                bottom = 'noBott'
                action = 'Right'
            if bottom == 'bott':
                action ='Up'
                if bump =='Bump':
                    action = 'Right'
                    bottom = 'bott'
                else:
                    action = 'Left'


            #elif lastAction == 'Left' and bump == 'Bump':
            #     leftEdge = 'leftEdge'
            #     action = 'Up'
            # elif leftEdge == 'leftEdge' and lastAction == 'Left' and ceil == 'noCeil':
            #     action = 'Up'
            # elif lastAction == 'Up' and bump == 'Bump':
            #     action = 'Right'
            #     ceil = 'ceil'
            #     leftEdge = 'noLeftEdge'
            # elif lastAction == 'Right' and bump == 'Bump':
            #     rightEdge = 'rightEdge'


        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt