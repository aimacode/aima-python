import random
from tkinter import *
import sys
import os.path
from agents import *
sys.path.append('./gui')
from xy_vacuum_environment import *

"""The main function."""
root = Tk()
root.title("Vacuum Environment")
root.geometry("420x440")
root.resizable(0, 0)
frame = Frame(root, bg='black')
reset_button = Button(frame, text='Reset', height=2,
                        width=6, padx=2, pady=2)
reset_button.pack(side='left')
next_button = Button(frame, text='Next', height=2,
                        width=6, padx=2, pady=2)
next_button.pack(side='left')
frame.pack(side='bottom')
env = Gui(root)
agt = ModelBasedVacuumAgent()
env.add_thing(agt, location=(3, 3))
next_button.config(command=env.update_env)
reset_button.config(command=lambda: env.reset_env(agt))
root.mainloop()

# Initialize Vacuum Environment
v = VacuumEnvironment(6,6)
#Get an agent
agent = ModelBasedVacuumAgent()
agent.direction = Direction(Direction.R)
v.add_thing(agent)
v.add_thing(Dirt(), location=(2,1))

# Check if things are added properly
assert len([x for x in v.things if isinstance(x, Wall)]) == 20
assert len([x for x in v.things if isinstance(x, Dirt)]) == 1

#Let the action begin!
assert v.percept(agent) == ("Clean", "None")
v.execute_action(agent, "Forward")
assert v.percept(agent) == ("Dirty", "None")
v.execute_action(agent, "TurnLeft")
v.execute_action(agent, "Forward")
assert v.percept(agent) == ("Dirty", "Bump")
v.execute_action(agent, "Suck")
assert  v.percept(agent) == ("Clean", "None")
old_performance = agent.performance
v.execute_action(agent, "NoOp")
assert old_performance == agent.performance
