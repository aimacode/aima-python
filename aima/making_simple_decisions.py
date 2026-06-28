"""Making Simple Decisions (Chapter 15/16)

The decision-theoretic algorithms — decision networks and the
information-gathering agent — are implemented in :mod:`aima.probability`,
alongside the Bayesian-network machinery they build on. They are re-exported
here under the chapter's name; :mod:`aima.probability` is their canonical home.
"""

from aima.probability import DecisionNetwork, InformationGatheringAgent

__all__ = ['DecisionNetwork', 'InformationGatheringAgent']
