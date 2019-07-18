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

class DirtyPerceptor(Perceptor):
    def percept(self, agent):
        return {'Dirty':len(self.env.find_at(Dirt, agent.location))>0}

class BumpPerceptor(Perceptor):
    def percept(self, agent):
        return {'Bump':len([o for o in self.env.objects_at(vector_add(agent.location, agent.heading)) if o.blocker])>0}

class RangePerceptor(Perceptor):
    default_r = 5  # default in the event that agent.sensor_r is not defined
    def percept(self, agent):
        if hasattr(agent,'sensor_r'):
            objs = self.env.objects_near(agent.location, agent.sensor_r)
        else:
            objs = self.env.objects_near(agent.location, self.default_r)
        return {'Objects':[(obj.__class__.__name__, (obj.location[0]-agent.location[0],obj.location[1]-agent.location[1]))
                for obj in objs]}

class GPSPerceptor(Perceptor):
    def percept(self, agent):
        return {'GPS':agent.location}

class CompassPerceptor(Perceptor):
    def percept(self, agent):
        return {'Compass':agent.heading}

class PerfectPerceptor(Perceptor):
    def percept(self, agent):
        return {'Objects':[(obj.__class__.__name__, obj.location) for obj in self.env.objects if isinstance(obj.location, tuple)]}
        # changed above because I think detecting if the dirt has a tuple as a location is more robust than detecting if the location is not an Agent