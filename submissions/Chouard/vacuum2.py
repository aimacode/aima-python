import agents as ag


def HW2Agent() -> object:

    # This agent will snake through the room to clean it.
    # It starts by heading for the top of the room. Then it snakes down
    # Once it reaches the bottom corner, it will go to the top as fast as possible and snake again.
    def program(percept):
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus, = program.oldPercepts[-1]
            lastAction = program.oldActions[-1]
            upOrDown, leftOrRight = program.snakeDirection
            if lastAction == 'NoOp':
                action = upOrDown
            elif lastStatus == 'Dirty':
                action = leftOrRight
            elif bump == 'None':
                if lastAction == 'Up':
                    # Go all the way up to the top of the room
                    action = 'Up'
                    program.snakeDirection = ('Up', leftOrRight)
                elif lastAction == 'Down':
                    action = 'Left' if leftOrRight == 'Right' else 'Right'
                    program.snakeDirection = (upOrDown, action)
                else:
                    action = lastAction
                    program.snakeDirection = (upOrDown, leftOrRight)
            else:
                if lastBump:
                    # we've bumped multiple times, so we're in a corner
                    if lastAction == 'Down':
                        action = 'Up'
                        program.snakeDirection = (action, leftOrRight)
                    elif lastAction == 'Up':
                        action = 'Left' if leftOrRight == 'Right' else 'Right'
                        program.snakeDirection = ('Down', action)
                    else:
                        action = upOrDown
                        program.snakeDirection = (action, leftOrRight)
                else:
                    action = upOrDown
                    program.snakeDirection = (action, leftOrRight)
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action
    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.snakeDirection = ('Up', 'Right')

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')
    return agt
