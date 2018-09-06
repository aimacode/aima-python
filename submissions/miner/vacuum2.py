import agents as ag

def HW2Agent() -> object:

    def program(percept):
        lastBump, lastStatus, = program.oldPercepts[-1]
        oldStatus = program.oldPercepts[0]
        lastAction = program.oldActions[-1]
        # print(lastBump, '= last bump ||', lastStatus, '= lastStatus ||', lastAction, '= lastActions')
        try:
            lastLastAction = program.oldActions[-2]
        except IndexError:
            lastLastAction = False

        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'
        else:
            # We have hit a bump
            if bump == 'Bump':
                # 1
                if lastAction == 'Down' and program.direction == 'Down':
                    print('#1')
                    action = 'Left'
                    program.direction = 'Left'
                # 2
                elif lastAction == 'Up' and lastLastAction == 'Left':
                    print('#2')
                    program.direction = 'Left'
                    action = 'Left'
                # 3
                elif lastAction == 'Left':
                    print('#3')
                    program.left = True
                    action = 'Right'
                # 4
                elif lastAction == 'Right':
                    print('#4')
                    program.right = True
                    action = 'Down'
                # 5
                elif lastAction == 'Down':
                    print('#5')
                    action = 'Up'
                # 6
                elif lastAction == 'Up':
                    print('#6')
                    program.direction = 'Left'
                    action = 'Left'
            else:
                # 7
                if program.left and lastAction == 'Left' and program.direction == '':
                    print('#7')
                    action = 'Down'
                    program.direction = 'Down'
                # 8
                elif program.direction == 'Left' and lastAction == 'Left' and lastLastAction == 'Up':
                    print('#8')
                    action = 'Down'
                    program.direction = 'Down'
                # 9
                elif program.direction == 'Left' and lastAction == 'Left' and lastLastAction == 'Down':
                    print('#9')
                    program.direction = 'Up'
                    action = 'Up'
                # 10
                elif lastAction == 'Down':
                    print('#10')
                    action = 'Down'
                # 11
                elif lastAction == 'Up':
                    print('#11')
                    action = 'Up'
                # 12
                elif lastAction == 'Right':
                    print('#12')
                    action = 'Right'
                # 13
                elif lastAction == 'Left':
                    print('#13')
                    action = 'Left'
                else:
                    action = 'Left'
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.direction = ''
    program.left = False
    program.right = False
    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt