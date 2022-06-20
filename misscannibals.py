from search import *


class MissCannibals(Problem):
    def __init__(self, M=3, C=3, goal=(0, 0, False)):
        initial = (M, C, True)
        self.M = M
        self.C = C
        super().__init__(initial, goal)

    # YOUR CODE GOES HERE
    def goal_test(self, state):

        return state == self.goal


    def is_valid(self, state):

        # checks that M and R are within range 0-initial
        if state[0] < 0 or state[0] > self.initial[0]:
            return False

        if state[1] < 0 or state[1] > self.initial[1]:
            return False

        if state[0] < state[1] and state[0] != 0:
            return False

        mr = self.initial[0] - state[0]
        cr = self.initial[1] - state[1]

        if mr < cr and mr != 0:
            return False

        return True

    def result(self, state, action):
        temp = list(state)

        # boat on left
        if temp[2]:
            temp[0] -= action.count('M')
            temp[1] -= action.count('C')

        # boat on right
        else:
            temp[0] += action.count('M')
            temp[1] += action.count('C')

        temp[2] = not temp[2]

        return tuple(temp)

    def actions(self, state):

        actions = []
        if self.is_valid(self.result(state, 'M')):
            actions.append('M')
        if self.is_valid(self.result(state, 'MM')):
            actions.append('MM')
        if self.is_valid(self.result(state, 'MC')):
            actions.append('MC')
        if self.is_valid(self.result(state, 'CC')):
            actions.append('CC')
        if self.is_valid(self.result(state, 'C')):
            actions.append('C')


        return actions




if __name__ == '__main__':
    mc = MissCannibals(3, 3)

    #print(mc.initial)


    # print(mc.actions((3, 2, True))) # Test your code as you develop! This should return  ['CC', 'C', 'M']

    path = depth_first_graph_search(mc).solution()
    print(path)
    path = breadth_first_graph_search(mc).solution()
    print(path)
