import agents as ag

def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean', 0, 0, 0,0, 'notFoundLeft', 'notFoundTop', 'notFoundRight', 'notFoundBottom', 'firstMove', 'middle')]
    oldActions = ['NoOp']


    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        lastBump, lastStatus, countLeft, countUp, countRight, countDown, foundLeft, foundTop, foundRight, foundBottom, firstMove, middle = oldPercepts[-1]
        # = oldPercepts[len[oldPercepts]-2]
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastAction = oldActions[-1]
            secondToLastAction = oldActions[len(oldActions)-2]

            if lastAction == "":
                action = ""

            elif (lastAction=="Down" and middle == "Begin" and bump=="Bump"):
                action = "Right"            #
            elif (middle == "stop"):
                action = "Down"
                foundBottom = "hasFoundBottom"
            elif (middle == 'begin' and lastAction == "Up" and bump == "Bump"):
                action = 'Right'
                foundTop = 'hasFoundTop'
                foundBottom = "notFoundBottom"
                middle = "stop"
            elif(foundBottom=="hasFoundBottom" and (lastAction=='Left' or 'Up')and bump=='Bump'):
                action='Up'
                middle ="begin"

            elif (lastAction == 'Left' and bump == "Bump") or (secondToLastAction == 'Left' and lastAction == 'Suck' and lastBump == "Bump") :
                action = 'Up'
                foundLeft = 'hasFoundLeft'
                countUp += 1
            elif (lastAction == 'Up' and bump =='Bump') or (secondToLastAction == 'Up' and lastAction == 'Suck' and lastBump == "Bump"):
                action = 'Right'
                countRight += 1
            elif (lastAction == 'Right' and bump == 'Bump') or (secondToLastAction == 'Right' and lastAction == 'Suck' and lastBump == "Bump"):
                action = 'Down'
                countDown += 1
            elif (lastAction == 'Down' and bump == 'Bump') or (secondToLastAction == 'Down' and lastAction == 'Suck' and lastBump == "Bump"):
                action = 'Left'
                foundBottom = "hasFoundBottom"
                countLeft += 1
            elif(lastAction=="Left"):
                action = 'Left'
            elif (lastAction == "Right"):
                action = 'Right'
                countRight += 1
            elif (lastAction == 'Down' or secondToLastAction=="Down" and foundTop != 'hasFoundTop' ):
                action = 'Down'
                countDown +=1

            else:
                action = 'Left'

        # print(countLeft, countUp, countRight, countDown, bump, lastBump)

            print(countDown)



            #     rest of first solution
            # if lastAction == 'Up':
            #     action = 'Right'
            # if secondToLastAction == 'Up':
            #     action = 'Down'
            # print(secondToLastAction)


        oldPercepts.append([bump, status, countLeft, countUp, countRight, countDown, foundLeft, foundTop, foundRight, foundBottom, firstMove, middle])
        oldActions.append(action)

        return action

    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']

    agt = ag.Agent(program)
    print (agt)
    return ag.Agent(program)