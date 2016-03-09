"""Reinforcement Learning (Chapter 21)
"""

if __name__ == "aimaPy.rl":
    from . utils import *
    from . import agents
else:
    from utils import *
    import agents


class PassiveADPAgent(agents.Agent):

    """Passive (non-learning) agent that uses adaptive dynamic programming
    on a given MDP and policy. [Fig. 21.2]"""
    NotImplemented


class PassiveTDAgent(agents.Agent):

    """Passive (non-learning) agent that uses temporal differences to learn
    utility estimates. [Fig. 21.4]"""
    NotImplemented
