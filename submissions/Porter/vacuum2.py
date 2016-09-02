import agents as ag
x=int



def HW2Agent() -> object:
    "An agent that keeps track of what locations are clean or dirty."
    oldPercepts = [('None', 'Clean')]
    oldActions = ['NoOp']



    def program(percept):
        "Same as ReflexVacuumAgent, except if everything is clean, do NoOp."
        bump, status = percept
        if status == 'Dirty':
            action = 'Suck'

        else:
            lastBump, lastStatus = oldPercepts[-1]
            if lastBump == 'None':
                action = 'Right'

            else:
                action = 'Left'


        oldPercepts.append(percept)
        oldActions.append(action)

        # V1.execute_action(self, agent, action)


        # x=1
        # while (x<10):
        #     x += 1
        #     ag.Environment.execute_action(self, agent, action)



        return action

        # x = 1
        # while (x < 10):
        #     program(percept)
        #     execute_action(program)
        #     x += 1




    return ag.Agent(program)

#Vacuum2Runner initializes the environment, dirt, and vacuum agent
#in vacuum2Runner, the execute_action method contains instructions for how the agent should interact with the environmetn based on the given action input
#vacuum2 receives a percept from vacuum2runner and decides which action to take
#there is no code that I can fine that passes the action from vacuum2 to vacuum2Runner to be executed
#I have commented out a few things I have tried.  I can't figure out how to get vacuum2 to call the execute_action method from vacuum2Runner
#logically all I need to do, as far as I can tell, is pass the action from vacuum2 to vacuum2Runner and then pass a new Percept to the program in vacuum2
