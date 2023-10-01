import search
import numpy as np
import io
import itertools


class FleetProblem (search.Problem ) :
    
    R=0 
    P=0
    V=0
    lugares=0
    requests=0 # [time, initial pos, final pos, people] 
    m_Time=0 #[PxP] matrix with time 
    '''
    state
    [r_status,v_status,time]
    r_status = 4 codigos:
                N - not ready: o pedido ainda nao foi feito
                R - ready : o pedido já foi feito
                P - PickedUp : as pessoas já foram apanhadas
                D - DorpedOff : o pedido já ficou resolvido
    v_status 
    [numero do veiculo,posição atual do veiculo,[0,0,0,0] cada numero representa o nº do request da pessoa, numero de lugares disponíveis]
    (-1) representa lugar vago         
    '''
    
    def __init__(self):
        
        request_status = ['N']*self.R
        vehicule_status = []
        if self.V != 0:
            for v in range(self.V):
                vehicule_status.append([v,0,[-1]*self.lugares[v],self.lugares[v]]) # [numero do veiculo,posição atual do veiculo,[0,0,0,0] cada numero representa o nº do request da pessoa]
                # (-1) representa lugar vago
            
        search.Problem.__init__(self, initial=[request_status, vehicule_status, 0]) 
        #[r_status,v_status, time]
        
    def find_max (self, initial, action):
        """Return the max time from all moves from inital positions to final positions"""
        time = []
        
        for i in enumerate(initial):
            time.append(self.m_Time[initial[i]][action[i][1]])
            
        return max(time)
    
    def check_Request_ready(self,state):
        """Updates request status from not ready to ready"""
        r_status = state
        
        if 'N' in state[1]:
            for r in enumerate(r_status[1]):
                if state[-1] > self.requests[r][0]:
                    r_status[1][r] = 'R'
                    
        return r_status    
    
    def result (self, state, action):
        """Return the state that results from exectuting given action in given state"""
        
        initial_positions = []
        temp_state = state
        '''
        action ---------------------------------------------> unfinished--------------------------------------------
        '''
        for v in self.V:
            initial_positions.append(state[v+1][1]) # get initial positions of vehicles
            
        for v in self.V:
            temp_state[v+1][1] = action[v][1] # update positions of vehicles
                
        temp_state[-1] += self.find_max(initial_positions,action) # update the time 
            
        temp_state = self.check_Request_ready(temp_state) # update from Not Ready to Ready
        
        #actions have been validated
        #update status from pickup
        #check if dropoff if yes update status!
        
        return temp_state
    
    def pickups_combs (self, pos, state):
        """Generates possible pickup combinations, can be impossible"""
        
        bools = [True, False]
        temp=[]
        actions=[]
        
        if 'R' in state[1]:
            
            bools = [[True if i == j else False for i in range(self.V)] for j in range(self.V)]
            bools.append([False]*self.V)
            combinations = list(itertools.product(bools, repeat = self.V))
            
            for i in combinations:
                temp=[]
                for a in enumerate(pos):
                    temp.append(pos[a],i[a])
                actions.append(temp)
        else:
            for i in pos:
                temp = [False]*self.R
                actions.append(i + temp)
        return actions 
    
    def action_val (self,action,state):
        """Verifies if action is valid or not"""
        initial_positions = []
        temp_state = state
        
        for v in self.V:
            initial_positions.append(state[v+1][1]) # get initial positions of vehicles
            
        for v in self.V:
            temp_state[v+1][1] = action[v][1] # update positions of vehicles
                
        temp_state[-1] += self.find_max(initial_positions,action) # update the time 
            
        temp_state = self.check_Request_ready(temp_state) # update from Not Ready to Ready
        
        for comand in enumerate(action): # action [final pos, R1,R2,R3,R4,... pickup yes or no]*self.V
            
            for i in enumerate(action[comand]):
                if i == 0 and action[comand][i] != self.requests[i-1][1]: # check if in position for pickup
                    return False
                elif i == True and temp_state[1][i-1] != 'R': # chek if it is available for pickup
                    return False
                elif temp_state[comand+1][-1] < self.requests[i-1][-1]: # check if space available in car
                    return False
                else:
                    temp_state[comand+1][-1] -= self.requests[i-1][-1] #u pdate fre space in car
        return True
    
    def actions (self, state):
        """Return actions that can be executed in the given state"""
        actions=[]
        
        # calculating all possibilities in car positions
        positions = range(self.P)
        combinations = list(itertools.product(positions, repeat = self.V))
        
        # actions have the final postions and the option to pickup or not
        # dropoff is automatic
        
        for comb in combinations:
                actions.append(self.pickups_combs(comb, state))
                
        for action in actions:
            if !self.action_val(action):
                actions.remove(action)
        '''
        action format:
        [final position,[T,F,list saying which requests to pickup or not]] * numero de veiculos
        para cada ação!
        '''
        return actions
    
    def goal_test(self, state):
        """Return True if the state is a goal"""
        #-------------------------------------------------------------unfinished-------------------------
        pass
    
    def load (self, fh):
        """Loads a problem from the file object fh"""
        while True :
            
            if self.P !=0 and self.V!=0:
                break
            
            line = fh.readline()
            
            if line[0] == '#':
                
                continue
            
            elif line[0] == 'P':
                
                newline = line[1:]
                self.P = int(newline)
                print(self.P)
                self.m_Time = np.zeros((self.P,self.P))
                
                for x in range(self.P-1):
                    
                    line = fh.readline()
                    split_str = line.split()
                    
                    for y in range(self.P-1-x):
                        
                        self.m_Time[x][y+1+x] = float(split_str[y])
                
                continue
            
            elif line[0] == 'R':
                
                newline = line[1:]
                self.R = int(newline)
                print(self.R)
                self.requests = np.zeros((self.R,4))
                
                for x in range(self.R):
                    
                    line = fh.readline()
                    split_str = line.split()
                    
                    for y in range(4):
                        
                        self.requests[x][y] = float(split_str[y])
                        
                request_status = ['N']*self.R 
                '''
                4 codigos:
                N - not ready: o pedido ainda nao foi feito
                R - ready : o pedido já foi feito
                P - PickedUp : as pessoas já foram apanhadas
                D - DorpedOff : o pedido já ficou resolvido
                '''
                continue
            
            elif line[0] == 'V':
                
                newline = line[1:]
                self.V = int(newline)
                self.lugares = [0]*self.V
                
                for x in range(self.V):
                    
                    line = fh.readline()
                    self.lugares[x] = int(line)
                    
                vehicule_status = []
                for v in range(self.V):
                    vehicule_status.append([v,0,[-1]*self.lugares[v],self.lugares[v]]) # [numero do veiculo,posição atual do veiculo,[0,0,0,0] cada numero representa o nº do request da pessoa]
                    # (-1) representa lugar vago, ultimo digito representa o numero de lugares disponíveis     
        search.Problem.__init__(self, initial=[request_status, vehicule_status, 0] ) #path_cost e parent são feitos automaticamente! ultimo elemento é o tempo
        
    
    def solve(self):
        """Calls the uninformed search algorithm chosen. Returns solutions in the specified format"""
        search.uniform_cost_search(FleetProblem())
        # tem de percorrer a solução e fazer print da solução no formato do 1ºassignment!!!!---------------------------unfinished--------------------
        return 'Test'
        
P = """
# this is a comment
V 1
2
R 5
10 1 2 1
15 1 3 1
16 2 3 1
20 2 1 1
25 1 2 2
P 4
20 30 40
   50 60
      70
"""    
    
def main():
    problem = FleetProblem()
    with io.StringIO(P) as fh:
        problem.load(fh)
    print(problem.solve())

        
if __name__=='__main__':
    main()
