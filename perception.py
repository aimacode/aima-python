from utils import vector_add
from objects import *

class Perceptor():
    def __init__(self, env):
        self.env = env

    def percept(self, agent):
        return None

    def object_percept(self, obj, agent):
        "Return the perception value of the object, for the specific agent."
        # Default value is a string representing the name of the class
        return obj.__class__.__name__

class BasicPerceptor(Perceptor):
    def percept(self, agent):
        return [self.object_percept(obj, agent)
                for obj in self.env.objects_at(agent)]

class DirtPerceptor(Perceptor):
    def percept(self, agent):
        return {'Dirty':len(self.env.find_at(Dirt, agent.location))>0}

class BumpPerceptor(Perceptor):
    def percept(self, agent):
        return {'Bump':len([o for o in self.env.objects_at(vector_add(agent.location, agent.heading)) if o.blocker])>0}

class RangePerceptor(Perceptor):
    r = 3  # TODO: how to handle multiple radii?
    def percept(self, agent):
        objs = self.env.objects_near(agent, self.r)
        return {'Objects':[(obj, (obj.location[0]-agent.location[0],obj.location[1]-agent.location[1]))
                for obj in objs]}

class GPSPerceptor(Perceptor):
    def percept(self, agent):
        return {'GPS':agent.location}
