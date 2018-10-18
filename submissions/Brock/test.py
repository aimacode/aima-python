import search



class Test:

    def __init__(self, name):
        self.name = name
    def state2String(self, myState):
        answer = ''
        for x in myState:
            for y in x:
                answer += y + ','
            answer = answer[:-1]
            answer += '|'
        return answer[:-1]


    def string2State(self, myString):
        state = myString.split('|')
        count = 0
        for x in state:
            state[count] = x.split(',')
            count = count + 1
        return state

SinglesInitState = [[0,0,3], [0,1,4], [1,0,3], [1,1,5]]
b = Test('Test')
print(b.state2String(SinglesInitState))
print(b.string2State(b.state2String(SinglesInitState)))
