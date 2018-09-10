import agents as ag
import random

def HW2Agent() -> object:

    def updatemap(Lastaction,currentPos,Bump):
        newPos = (0,0)
        if Lastaction == 'Left' and Bump == 'Bump':
            #program.map.append[currentPos][0]= 1
            program.bounds[0] = currentPos[0]
            print(str(program.bounds) + 'Left Bump')
            program.hasBumped[0] = True
            #program.bumpCount = program.bumpCount + 1
            #program.currentPos= (currentPos[0]-1, currentPos[1])
            print(program.currentPos)
        elif Lastaction == 'Up' and Bump == 'Bump':
           # program.map[currentPos][1] = 1
            program.bounds[1]=currentPos[1]
            program.hasBumped[1] = True
            program.bumpCount = program.bumpCount + 1
            print(str(program.bounds) + 'Up Bump')
            #program.currentPos = (currentPos[0], currentPos[1]+1)
            print(program.currentPos)
        elif Lastaction == 'Right' and Bump == 'Bump':
           # program.map[currentPos][2] = 1
            program.bumpCount = program.bumpCount + 1
            program.bounds[2]=currentPos[0]
            program.hasBumped[2] = True
            print(str(program.bounds) + 'Right Bump')
            #program.currentPos=(currentPos[0]+1,currentPos[1])
            print(program.currentPos)
        elif Lastaction == 'Down' and Bump == 'Bump':
           # program.map[currentPos][3] = 1
            program.bumpCount = program.bumpCount + 1
            program.bounds[3] = currentPos[1]
            program.hasBumped[0] = True
            print(str(program.bounds) + 'Down Bump')
            #program.currentPos = (currentPos[0], currentPos[1]-1)
            #print('CurrentPos:' + str(program.currentPos))



        elif Lastaction == 'Left' and Bump != 'Bump':
             program.currentPos = (currentPos[0] - 1, currentPos[1])
        elif Lastaction == 'Up' and Bump != 'Bump':
           # program.map[currentPos][1] = 0
           # program.bounds[1]=currentPos[1
             program.currentPos = (currentPos[0], currentPos[1]+1)
        elif Lastaction == 'Right' and Bump != 'Bump':
            #program.map[currentPos][2] = 0
           # program.bounds[2]=currentPos[2]
             program.currentPos = (currentPos[0] + 1, currentPos[1])
        elif Lastaction == 'Down' and Bump != 'Bump':
            #program.map[currentPos][3] = 0
            #program.bounds[3] = currentPos[3]
             program.currentPos = (currentPos[0], currentPos[1] - 1)

        print('CurrentPos out:' + str(program.currentPos))
        print('BumpCount: '+ str(program.bumpCount))



    #def evalMove():









    def program(percept):
        bump, status = percept
        lastAction = program.oldActions[-1]
        """""
        if bump == 'Bump':
            program.bumpCount = program.bumpCount + 1
        if program.currentPos[0] == program.bounds[0] or program.currentPos[0] == program.bounds[2] or program.currentPos[1] == program.bounds[1] or program.currentPos[1]== program.bounds[3]:
            program.bumpCount = program.bumpCount + 1
        if program.bumpCount == 4:
            program.hasBounds = True
            program.bounds[0]= program.bounds[0] + 1
            program.bounds[1] = program.bounds[1]-1
            program.bounds[2] =program.bounds[2]-1
            program.bounds[3] =program.bounds[3]+1
            program.bumpCount =0
        """

        #if program.bounds[0] != -100 and program.bounds[1]!= 100 and program.bounds[2] != 100 and program.bounds[3] != -100:
         #   program.hasBounds = True
        """
        if program.currentPos[0] == program.bounds[0] and program.hasBounds == True or program.currentPos[0] == program.bounds[2] and program.hasBounds == True or program.currentPos[1]== program.bounds[1] and program.hasBounds == True or program.currentPos[1] == program.bounds[3] and program.hasBounds == True:
            program.bumpCount= program.bumpCount +1
            print('Bound Bump')
        if program.bumpCount % 4 == 0:
            program.bounds[0] = program.bounds[0] + 1
            program.bounds[1] = program.bounds[1] - 1
            program.bounds[2] = program.bounds[2] - 1
            program.bounds[3] = program.bounds[3] + 1
        """
        if status == 'Dirty':
            action = 'Suck'
        else:
            lastBump, lastStatus,  = program.oldPercepts[-1]

            if bump == 'None' and program.currentPos[0] <= program.bounds[2]:
                action = 'Right'
            if bump == 'Bump' and program.currentPos[1] <= program.bounds[1] or lastAction == 'Up' and program.currentPos[1] <= program.bounds[1]:
                action = 'Up'
            if bump == 'Bump' and lastAction == 'Up' and program.currentPos[0]>= program.bounds[0] or lastAction == 'Left' and program.currentPos[0]>= program.bounds[0]:
                action = 'Left'
            if bump =='Bump' and lastAction== 'Left' and program.currentPos[1]>= program.bounds[3] or lastAction == 'Down' and program.currentPos[1]>= program.bounds[3]:
                action = 'Down'
            if bump =='Bump' and lastAction =='Down':
                action = 'Right'

            """"
            if bump == 'None' and program.currentPos[0] <= program.bounds[2] and program.hasBounds==True:
                action = 'Right'
            if program.currentPos[1] <= program.bounds[1] or lastAction == 'Up' and program.currentPos[1] <= program.bounds[1] and program.hasBounds==True:
                action = 'Up'
            if lastAction == 'Up' and program.currentPos[0]>= program.bounds[0] or lastAction == 'Left' and program.currentPos[0]>= program.bounds[0] and program.hasBounds==True:
                action = 'Left'
            if lastAction== 'Left' and program.currentPos[1]>= program.bounds[3] or lastAction == 'Down' and program.currentPos[1]>= program.bounds[3] and program.hasBounds==True:
                action = 'Down'
            if lastAction =='Down' and program.hasBounds==True:
                action = 'Right'
            """

            """""
            if program.hasBounds == True and lastAction=='Right' and program.currentPos[0]==program.bounds[1]:
                action == 'Up'
            if program.hasBounds == True and lastAction=='Up' and program.currentPos[1]< program.bounds[2] or program.hasBounds==True and lastAction == 'Left' and program.currentPos[0] >= program.bounds[0]:
                action == 'Left'
            if program.hasBounds == True and lastAction=='Left' or program.hasBounds==True and lastAction == 'Down' and program.currentPos >= program.bounds[3]:
                action == 'Down'
            """








        #updatemap(lastAction, program.currentPos, program.oldPercepts[-1][0])
        program.oldPercepts.append(percept)
        program.oldActions.append(action)
        print('Bump in:' + bump)
        print('CurrentPos in:' + str(program.currentPos))
        print('Last Action:' + lastAction)
        updatemap(lastAction, program.currentPos, bump)

        #print(program.bounds)

        return action

    # assign static variables here
    program.oldPercepts = [('None', 'Clean')]
    program.oldActions = ['NoOp']
    program.bumpCount = 0
    program.hasBumped=[False,False,False,False]
    program.hasBounds= False
    program.currentPos=(0,0)
    program.bounds = [-100,100,100,-100]
    print(program.bounds)

    agt = ag.Agent(program)
    # assign class attributes here:
    # agt.direction = ag.Direction('left')

    return agt

