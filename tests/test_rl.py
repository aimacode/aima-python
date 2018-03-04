import pytest

from rl import *
from mdp import sequential_decision_environment


north = (0, 1)
south = (0,-1)
west = (-1, 0)
east = (1, 0)

policy = {
    (0, 2): east,  (1, 2): east,  (2, 2): east,   (3, 2): None,
    (0, 1): north,                (2, 1): north,  (3, 1): None,
    (0, 0): north, (1, 0): west,  (2, 0): west,   (3, 0): west, 
}

def test_PassiveDUEAgent():
	agent = PassiveDUEAgent(policy, sequential_decision_environment)
	for i in range(200):
		run_single_trial(agent,sequential_decision_environment)
		agent.estimate_U()
	# Agent does not always produce same results.
	# Check if results are good enough.
	#print(agent.U[(0, 0)], agent.U[(0,1)], agent.U[(1,0)])
	assert agent.U[(0, 0)] > 0.15 # In reality around 0.3
	assert agent.U[(0, 1)] > 0.15 # In reality around 0.4
	assert agent.U[(1, 0)] > 0 # In reality around 0.2

def test_PassiveADPAgent():
	agent = PassiveADPAgent(policy, sequential_decision_environment)
	for i in range(100):
		run_single_trial(agent,sequential_decision_environment)
	
	# Agent does not always produce same results.
	# Check if results are good enough.
	#print(agent.U[(0, 0)], agent.U[(0,1)], agent.U[(1,0)])
	assert agent.U[(0, 0)] > 0.15 # In reality around 0.3
	assert agent.U[(0, 1)] > 0.15 # In reality around 0.4
	assert agent.U[(1, 0)] > 0 # In reality around 0.2



def test_PassiveTDAgent():
	agent = PassiveTDAgent(policy, sequential_decision_environment, alpha=lambda n: 60./(59+n))
	for i in range(200):
		run_single_trial(agent,sequential_decision_environment)
	
	# Agent does not always produce same results.
	# Check if results are good enough.
	assert agent.U[(0, 0)] > 0.15 # In reality around 0.3
	assert agent.U[(0, 1)] > 0.15 # In reality around 0.35
	assert agent.U[(1, 0)] > 0.15 # In reality around 0.25


def test_QLearning():
	q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, 
							 alpha=lambda n: 60./(59+n))

	for i in range(200):
		run_single_trial(q_agent,sequential_decision_environment)

	# Agent does not always produce same results.
	# Check if results are good enough.
	assert q_agent.Q[((0, 1), (0, 1))] >= -0.5 # In reality around 0.1
	assert q_agent.Q[((1, 0), (0, -1))] <= 0.5 # In reality around -0.1
