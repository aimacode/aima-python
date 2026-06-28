# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %run bootstrap.ipynb

# %% [markdown] pycharm={}
# # Propositional Logic
# ---
# # Improving Boolean Satisfiability Algorithms
#
# ## Introduction
# A propositional formula $\Phi$ in *Conjunctive Normal Form* (CNF) is a conjunction of clauses $\omega_j$, with $j \in \{1,...,m\}$. Each clause being a disjunction of literals and each literal being either a positive ($x_i$) or a negative ($\lnot{x_i}$) propositional variable, with $i \in \{1,...,n\}$. By denoting with $[\lnot]$ the possible presence of $\lnot$, we can formally define $\Phi$ as:
#
# $$\bigwedge_{j = 1,...,m}\bigg(\bigvee_{i \in \omega_j} [\lnot] x_i\bigg)$$
#
# The ***Boolean Satisfiability Problem*** (SAT) consists in determining whether there exists a truth assignment in $\{0, 1\}$ (or equivalently in $\{True,False\}$) for the variables in $\Phi$.

# %%
from aima.logic import *
from aima.utils import open_data

# %% [markdown]
# ## DPLL with Branching Heuristics
# The ***Davis-Putnam-Logemann-Loveland*** (DPLL) algorithm is a *complete* (will answer SAT if a solution exists) and *sound* (it will not answer SAT for an unsatisfiable formula) procedue that combines *backtracking search* and *deduction* to decide satisfiability of propositional logic formula in CNF. At each search step a variable and a propositional value are selected for branching purposes. With each branching step, two values can be assigned to a variable, either 0 or 1. Branching corresponds to assigning the chosen value to the chosen variable. Afterwards, the logical consequences of each branching step are evaluated. Each time an unsatisfied clause (ie a *conflict*) is identified, backtracking is executed. Backtracking corresponds to undoing branching steps until an unflipped branch is reached. When both values have been assigned to the selected variable at a branching step, backtracking will undo this branching step. If for the first branching step both values have been considered, and backtracking undoes this first branching step, then the CNF formula can be declared unsatisfiable. This kind of backtracking is called *chronological backtracking*.
#
# Essentially, `DPLL` is a backtracking depth-first search through partial truth assignments which uses a *splitting rule* to replaces the original problem with two smaller subproblems, whereas the original Davis-Putnam procedure uses a variable elimination rule which replaces the original problem with one larger subproblem. Over the years, many heuristics have been proposed in choosing the splitting variable (which variable should be assigned a truth value next).
#
# Search algorithms that are based on a predetermined order of search are called static algorithms, whereas the ones that select them at the runtime are called dynamic. The first SAT search algorithm, the Davis-Putnam procedure is a static algorithm. Static search algorithms are usually very slow in practice and for this reason perform worse than dynamic search algorithms. However, dynamic search algorithms are much harder to design, since they require a heuristic for predetermining the order of search. The fundamental element of a heuristic is a branching strategy for selecting the next branching literal. This must not require a lot of time to compute and yet it must provide a powerful insight into the problem instance.
#
# Two basic heuristics are applied to this algorithm with the potential of cutting the search space in half. These are the *pure literal rule* and the *unit clause rule*.
# - the *pure literal* rule is applied whenever a variable appears with a single polarity in all the unsatisfied clauses. In this case, assigning a truth value to the variable so that all the involved clauses are satisfied is highly effective in the search;
# - if some variable occurs in the current formula in a clause of length 1 then the *unit clause* rule is applied. Here, the literal is selected and a truth value so the respective clause is satisfied is assigned. The iterative application of the unit rule is commonly reffered to as *Boolean Constraint Propagation* (BCP).

# %%
# %psource dpll_satisfiable

# %%
# %psource dpll

# %% [markdown]
# Each of these branching heuristics was applied only after the *pure literal* and the *unit clause* heuristic failed in selecting a splitting variable.

# %% [markdown]
# ### MOMs

# %% [markdown]
# MOMs heuristics are simple, efficient and easy to implement. The goal of these heuristics is to prefer the literal having ***Maximum number of Occurences in the Minimum length clauses***. Intuitively, the literals belonging to the minimum length clauses are the most constrained literals in the formula. Branching on them will maximize the effect of BCP and the likelihood of hitting a dead end early in the search tree (for unsatisfiable problems). Conversely, in the case of satisfiable formulas, branching on a highly constrained variable early in the tree will also increase the likelihood of a correct assignment of the remained open literals.
# The MOMs heuristics main disadvatage is that their effectiveness highly depends on the problem instance. It is easy to see that the ideal setting for these heuristics is considering the unsatisfied binary clauses.

# %%
# %psource min_clauses

# %%
# %psource moms

# %% [markdown]
# Over the years, many types of MOMs heuristics have been proposed.
#
# ***MOMSf*** choose the variable $x$ with a maximize the function:
#
# $$[f(x) + f(\lnot{x})] * 2^k + f(x) * f(\lnot{x})$$
#
# where $f(x)$ is the number of occurrences of $x$ in the smallest unknown clauses, k is a parameter.

# %%
# %psource momsf

# %% [markdown]
# ***Freeman’s POSIT*** <a name="ref-1"/>[[1]](#cite-freeman1995improvements) version counts both the number of positive $x$ and negative $\lnot{x}$ occurrences of a given variable $x$.

# %%
# %psource posit

# %% [markdown]
# ***Zabih and McAllester’s*** <a name="ref-2"/>[[2]](#cite-zabih1988rearrangement) version of the heuristic counts the negative occurrences $\lnot{x}$ of each given variable $x$.

# %%
# %psource zm

# %% [markdown]
# ### DLIS & DLCS

# %% [markdown]
# Literal count heuristics count the number of unresolved clauses in which a given variable $x$ appears as a positive literal, $C_P$ , and as negative literal, $C_N$. These two numbers an either be onsidered individually or ombined. 
#
# ***Dynamic Largest Individual Sum*** heuristic considers the values $C_P$ and $C_N$ separately: select the variable with the largest individual value and assign to it value true if $C_P \geq C_N$, value false otherwise.

# %%
# %psource dlis

# %% [markdown]
# ***Dynamic Largest Combined Sum*** considers the values $C_P$ and $C_N$ combined: select the variable with the largest sum $C_P + C_N$ and assign to it value true if $C_P \geq C_N$, value false otherwise.

# %%
# %psource dlcs

# %% [markdown]
# ### JW & JW2

# %% [markdown]
# Two branching heuristics were proposed by ***Jeroslow and Wang*** in <a name="ref-3"/>[[3]](#cite-jeroslow1990solving).
#
# The *one-sided Jeroslow and Wang*’s heuristic compute:
#
# $$J(l) = \sum_{l \in \omega \land \omega \in \phi} 2^{-|\omega|}$$
#
# and selects the assignment that satisfies the literal with the largest value $J(l)$.

# %%
# %psource jw

# %% [markdown]
# The *two-sided Jeroslow and Wang*’s heuristic identifies the variable $x$ with the largest sum $J(x) + J(\lnot{x})$, and assigns to $x$ value true, if $J(x) \geq J(\lnot{x})$, and value false otherwise.

# %%
# %psource jw2

# %% [markdown]
# ## CDCL with 1UIP Learning Scheme, 2WL Lazy Data Structure, VSIDS Branching Heuristic & Restarts
#
# The ***Conflict-Driven Clause Learning*** (CDCL) solver is an evolution of the *DPLL* algorithm that involves a number of additional key techniques:
#
# - non-chronological backtracking or *backjumping*;
# - *learning* new *clauses* from conflicts during search by exploiting its structure;
# - using *lazy data structures* for storing clauses;
# - *branching heuristics* with low computational overhead and which receive feedback from search;
# - periodically *restarting* search.
#
# The first difference between a DPLL solver and a CDCL solver is the introduction of the *non-chronological backtracking* or *backjumping* when a conflict is identified. This requires an iterative implementation of the algorithm because only if the backtrack stack is managed explicitly it is possible to backtrack more than one level.

# %%
# %psource cdcl_satisfiable

# %% [markdown]
# ### Clause Learning with 1UIP Scheme

# %% [markdown]
# The second important difference between a DPLL solver and a CDCL solver is that the information about a conflict is reused by learning: if a conflicting clause is found, the solver derive a new clause from the conflict and add it to the clauses database.
#
# Whenever a conflict is identified due to unit propagation, a conflict analysis procedure is invoked. As a result, one or more new clauses are learnt, and a backtracking decision level is computed. The conflict analysis procedure analyzes the structure of unit propagation and decides which literals to include in the learnt clause. The decision levels associated with assigned variables define a partial order of the variables. Starting from a given unsatisfied clause (represented in the implication graph with vertex $\kappa$), the conflict analysis procedure visits variables implied at the most recent decision level (ie the current largest decision level), identifies the antecedents of visited variables, and keeps from the antecedents the literals assigned at decision levels less than the most recent decision level. The clause learning procedure used in the CDCL can be defined by a sequence of selective resolution operations, that at each step yields a new temporary clause. This process is repeated until the most recent decision variable is visited.
#
# The structure of implied assignments induced by unit propagation is a key aspect of the clause learning procedure. Moreover, the idea of exploiting the structure induced by unit propagation was further exploited with ***Unit Implication Points*** (UIPs). A UIP is a *dominator* in the implication graph and represents an alternative decision assignment at the current decision level that results in the same conflict. The main motivation for identifying UIPs is to reduce the size of learnt clauses. Clause learning could potentially stop at any UIP, being quite straightforward to conclude that the set of literals of a clause learnt at the first UIP has clear advantages. Considering the largest decision level of the literals of the clause learnt at each UIP, the clause learnt at the first UIP is guaranteed to contain the smallest one. This guarantees the highest backtrack jump in the search tree.

# %%
# %psource conflict_analysis

# %%
# %psource pl_binary_resolution

# %%
# %psource backjump

# %% [markdown]
# ### 2WL Lazy Data Structure

# %% [markdown]
# Implementation issues for SAT solvers include the design of suitable data structures for storing clauses. The implemented data structures dictate the way BCP are implemented and have a significant impact on the run time performance of the SAT solver. Recent state-of-the-art SAT solvers are characterized by using very efficient data structures, intended to reduce the CPU time required per each node in the search tree. Conversely, traditional SAT data structures are accurate, meaning that is possible to know exactly the value of each literal in the clause. Examples of the most recent SAT data structures, which are not accurate and therefore are called lazy, include the watched literals used in Chaff .
#
# The more recent Chaff SAT solver <a name="ref-4"/>[[4]](#cite-moskewicz2001chaff) proposed a new data structure, the ***2 Watched Literals*** (2WL), in which two references are associated with each clause. There is no order relation between the two references, allowing the references to move in any direction. The lack of order between the two references has the key advantage that no literal references need to be updated when backtracking takes place. In contrast, unit or unsatisfied clauses are identified only after traversing all the clauses’ literals; a clear drawback. The two watched literal pointers are undifferentiated as there is no order relation. Again, each time one literal pointed by one of these pointers is assigned, the pointer has to move inwards. These pointers may move in both directions. This causes the whole clause to be traversed when the clause becomes unit. In addition, no references have to be kept to the just assigned literals, since pointers do not move when backtracking.

# %%
# %psource unit_propagation

# %%
# %psource TwoWLClauseDatabase

# %% [markdown]
# ### VSIDS Branching Heuristic

# %% [markdown]
# The early branching heuristics made use of all the information available from the data structures, namely the number of satisfied, unsatisfied and unassigned literals. These heuristics are updated during the search and also take into account the clauses that are learnt. 
#
# More recently, a different kind of variable selection heuristic, referred to as ***Variable State Independent Decaying Sum*** (VSIDS), has been proposed by Chaff authors in <a name="ref-4"/>[[4]](#cite-moskewicz2001chaff). One of the reasons for proposing this new heuristic was the introduction of lazy data structures, where the knowledge of the dynamic size of a clause is not accurate. Hence, the heuristics described above cannot be used. VSIDS selects the literal that appears most frequently over all the clauses, which means that one counter is required for each one of the literals. Initially, all counters are set to zero. During the search, the metrics only have to be updated when a new recorded clause is created. More than to develop an accurate heuristic, the motivation has been to design a fast (but dynamically adapting) heuristic. In fact, one of the key properties of this strategy is the very low overhead, due to being independent of the variable state.

# %% pycharm={}
# %psource assign_decision_literal

# %% [markdown]
# ### Restarts

# %% [markdown]
# Solving NP-complete problems, such as SAT, naturally leads to heavy-tailed run times. To deal with this, SAT solvers frequently restart their search to avoid the runs that take disproportionately longer. What restarting here means is that the solver unsets all variables and starts the search using different variable assignment order.
#
# While at first glance it might seem that restarts should be rare and become rarer as the solving has been going on for longer, so that the SAT solver can actually finish solving the problem, the trend has been towards more aggressive (frequent) restarts.
#
# The reason why frequent restarts help solve problems faster is that while the solver does forget all current variable assignments, it does keep some information, specifically it keeps learnt clauses, effectively sampling the search space, and it keeps the last assigned truth value of each variable, assigning them the same value the next time they are picked to be assigned.

# %% [markdown]
# #### Luby

# %% [markdown]
# In this strategy, the number of conflicts between 2 restarts is based on the *Luby* sequence. The *Luby* restart sequence is interesting in that it was proven to be optimal restart strategy for randomized search algorithms where the runs do not share information. While this is not true for SAT solving, as shown in <a name="ref-5"/>[[5]](cite-haim2014towards) and <a name="ref-6"/>[[6]](cite-huang2007effect), *Luby* restarts have been quite successful anyway.
#
# The exact description of *Luby* restarts is that the $ith$ restart happens after $u \cdot Luby(i)$ conflicts, where $u$ is a constant and $Luby(i)$ is defined as:
#
# $$Luby(i) = \begin{cases} 
#       2^{k-1} & i = 2^k - 1 \\
#       Luby(i - 2^{k-1} + 1) & 2^{k-1} \leq i < 2^k - 1
#    \end{cases}
# $$
#
# A less exact but more intuitive description of the *Luby* sequence is that all numbers in it are powers of two, and after a number is seen for the second time, the next number is twice as big. The following are the first 16 numbers in the sequence:
#
# $$ (1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,1,...) $$
#
# From the above, we can see that this restart strategy tends towards frequent restarts, but some runs are kept running for much longer, and there is no upper limit on the longest possible time between two restarts.

# %%
# %psource luby

# %% [markdown]
# #### Glucose

# %% [markdown]
# Glucose restarts were popularized by the *Glucose* solver, and it is an extremely aggressive, dynamic restart strategy. The idea behind it and described in <a name="ref-7"/>[[7]](cite-audemard2012refining) is that instead of waiting for a fixed amount of conflicts, we restart when the last couple of learnt clauses are, on average, bad.
#
# A bit more precisely, if there were at least $X$ conflicts (and thus $X$ learnt clauses) since the last restart, and the average *Literal Block Distance* (LBD) (a criterion to evaluate the quality of learnt clauses as shown in <a name="ref-8"/>[[8]](#cite-audemard2009predicting) of the last $X$ learnt clauses was at least $K$ times higher than the average LBD of all learnt clauses, it is time for another restart. Parameters $X$ and $K$ can be tweaked to achieve different restart frequency, and they are usually kept quite small.

# %%
# %psource glucose

# %% [markdown] pycharm={}
# ## Experimental Results

# %%
from aima.csp import *

# %% [markdown]
# ### Australia

# %% [markdown]
# #### CSP

# %%
australia_csp = MapColoringCSP(list('RGB'), """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """)

# %%
# %time _, checks = AC3b(australia_csp, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP needs {checks} consistency-checks'

# %%
# %time backtracking_search(australia_csp, select_unassigned_variable=mrv, inference=forward_checking)

# %% [markdown]
# #### SAT

# %%
australia_sat = MapColoringSAT(list('RGB'), """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """)

# %% [markdown]
# ##### DPLL

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=no_branching_heuristic)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=moms)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=momsf)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=posit)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=zm)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=dlis)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=dlcs)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=jw)

# %%
# %time model = dpll_satisfiable(australia_sat, branching_heuristic=jw2)

# %% [markdown]
# ##### CDCL

# %%
# %time model = cdcl_satisfiable(australia_sat)

# %%
{var for var, val in model.items() if val}

# %% [markdown]
# ### France

# %% [markdown]
# #### CSP

# %%
france_csp = MapColoringCSP(list('RGBY'),
                            """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""")

# %%
# %time _, checks = AC3b(france_csp, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP needs {checks} consistency-checks'

# %%
# %time backtracking_search(france_csp, select_unassigned_variable=mrv, inference=forward_checking)

# %% [markdown]
# #### SAT

# %%
france_sat = MapColoringSAT(list('RGBY'),
                            """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""")

# %% [markdown]
# ##### DPLL

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=no_branching_heuristic)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=moms)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=momsf)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=posit)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=zm)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=dlis)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=dlcs)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=jw)

# %%
# %time model = dpll_satisfiable(france_sat, branching_heuristic=jw2)

# %% [markdown]
# ##### CDCL

# %%
# %time model = cdcl_satisfiable(france_sat)

# %%
{var for var, val in model.items() if val}

# %% [markdown]
# ### USA

# %% [markdown]
# #### CSP

# %%
usa_csp = MapColoringCSP(list('RGBY'),
                         """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """)

# %%
# %time _, checks = AC3b(usa_csp, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP needs {checks} consistency-checks'

# %%
# %time backtracking_search(usa_csp, select_unassigned_variable=mrv, inference=forward_checking)

# %% [markdown]
# #### SAT

# %%
usa_sat = MapColoringSAT(list('RGBY'),
                         """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """)

# %% [markdown]
# ##### DPLL

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=no_branching_heuristic)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=moms)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=momsf)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=posit)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=zm)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=dlis)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=dlcs)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=jw)

# %%
# %time model = dpll_satisfiable(usa_sat, branching_heuristic=jw2)

# %% [markdown]
# ##### CDCL

# %%
# %time model = cdcl_satisfiable(usa_sat)

# %%
{var for var, val in model.items() if val}

# %% [markdown]
# ### Zebra Puzzle

# %% [markdown]
# #### CSP

# %%
zebra_csp = Zebra()

# %%
zebra_csp.display(zebra_csp.infer_assignment())

# %%
# %time _, checks = AC3b(zebra_csp, arc_heuristic=dom_j_up)
f'AC3b with DOM J UP needs {checks} consistency-checks'

# %%
zebra_csp.display(zebra_csp.infer_assignment())

# %%
# %time backtracking_search(zebra_csp, select_unassigned_variable=mrv, inference=forward_checking)

# %% [markdown]
# #### SAT

# %%
zebra_sat = associate('&', map(to_cnf, map(expr, filter(lambda line: line[0] not in ('c', 'p'), open_data('zebra.cnf').read().splitlines()))))

# %% [markdown]
# ##### DPLL

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=no_branching_heuristic)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=moms)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=momsf)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=posit)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=zm)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=dlis)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=dlcs)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=jw)

# %%
# %time model = dpll_satisfiable(zebra_sat, branching_heuristic=jw2)

# %% [markdown]
# ##### CDCL

# %% pycharm={}
# %time model = cdcl_satisfiable(zebra_sat)

# %%
{var for var, val in model.items() if val and var.op.startswith(('Englishman', 'Japanese', 'Norwegian', 'Spaniard', 'Ukrainian'))}

# %% [markdown]
# ## References
#
# <a name="cite-freeman1995improvements"/><sup>[[1]](#ref-1) </sup>Freeman, Jon William. 1995. _Improvements to propositional satisfiability search algorithms_.
#
# <a name="cite-zabih1988rearrangement"/><sup>[[2]](#ref-2) </sup>Zabih, Ramin and McAllester, David A. 1988. _A Rearrangement Search Strategy for Determining Propositional Satisfiability_.
#
# <a name="cite-jeroslow1990solving"/><sup>[[3]](#ref-3) </sup>Jeroslow, Robert G and Wang, Jinchang. 1990. _Solving propositional satisfiability problems_.
#
# <a name="cite-moskewicz2001chaff"/><sup>[[4]](#ref-4) </sup>Moskewicz, Matthew W and Madigan, Conor F and Zhao, Ying and Zhang, Lintao and Malik, Sharad. 2001. _Chaff: Engineering an efficient SAT solver_.
#
# <a name="cite-haim2014towards"/><sup>[[5]](#ref-5) </sup>Haim, Shai and Heule, Marijn. 2014. _Towards ultra rapid restarts_.
#
# <a name="cite-huang2007effect"/><sup>[[6]](#ref-6) </sup>Huang, Jinbo and others. 2007. _The Effect of Restarts on the Efficiency of Clause Learning_.
#
# <a name="cite-audemard2012refining"/><sup>[[7]](#ref-7) </sup>Audemard, Gilles and Simon, Laurent. 2012. _Refining restarts strategies for SAT and UNSAT_.
#
# <a name="cite-audemard2009predicting"/><sup>[[8]](#ref-8) </sup>Audemard, Gilles and Simon, Laurent. 2009. _Predicting learnt clauses quality in modern SAT solvers_.
