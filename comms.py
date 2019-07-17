
from agents import Agent


class Communicator():
    def __init__(self, env):
        self.env = env

    def get_comms_network(self, toAgent):
        '''return the list of Agents that the toAgent is able to communicate with'''
        range = 5
        [o for o in self.env.objects_near(toAgent.location, range) if isinstance(o, Agent)]

    def communicate(self, message, toAgent, fromAgent):
        '''communicate a message from the fromAgent to the toAgent'''
        return None

    def run_comms(self, agents):
        pass

class BroadcastCommunicator(Communicator):


    def communicate(self, message, toAgent, fromAgent):
        pass