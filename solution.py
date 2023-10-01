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
    pickups=[] # list of pickup locations
    def __init__(self):
        
        request_status = ['N']*self.R
        vehicule_status = []
        if self.V != 0:
            for v in range(self.V):
                vehicule_status.append([v,0,[-1]*self.lugares[v]]) # [numero do veiculo,posição atual do veiculo,[0,0,0,0] cada numero representa o nº do request da pessoa]
                # (-1) representa lugar vago
            
        search.Problem.__init__(self, initial=[request_status, vehicule_status, 0]) 
        #[r_status,v_status, time]
        
    def find_max (self, initial, final):
        """Return the max time from all moves from inital positions to final positions"""
        time = []
        
        for i in range(len(initial)):
            time.append(self.m_Time[initial[i]][final[i]])
            
        return max(time)
    
    def check_Request_ready(self,state):
        """Updates request status from not ready to ready"""
        r_status= state
        
        if 'N' in state[1]:
            for r in enumerate(r_status[1]):
                if state[-1] > self.requests[r][0]:
                    r_status[1][r] = 'R'
                    
        return r_status    
    
    def result (self, state, action):
        """Return the state that results from exectuting given action in given state"""
        
        initial_positions = []
        new_states = []
        temp_state = state
        '''
        action list is going to be redone ------> missing redo here
        '''
        for v in self.V:
            initial_positions.append(state[v+1][1]) # get initial positions of vehicles
            
        for v in self.V:
            temp_state[v+1][1] = action[v] # update positions of vehicles
                
            temp_state[-1] = self.find_max(initial_positions,action) # update the time 
            
            temp_state = self.check_Request_ready(temp_state) # update from Not Ready to Ready
            
            if 'R' in temp_state[1]:
                
                for r in enumerate(temp_state[1]):
                    if temp_state[1][r] == 'R' and self.requests[r][1] in action:
                        # change to read if action says pick up or not
            
            # if can pickup do it
            # if multiple cars for pick up, add states for all possibilites
            # if can dropoff do it
        pass
    
    def pickups_combs (self, action):
        """Generates possible pickup combinations"""
        #TO DO generate possible pickup true/false combinations----------------------------------------------------
        pass
    
    def actions (self, state):
        """Return actions that can be executed in the given state"""
        actions=[]
        
        # calculating all possibilities in car positions
        positions = range(self.P)
        combinations = list(itertools.product(positions, repeat = self.V))
        
        # actions have the final postions and the option to pickup or not
        # dropoff is automatic
        if 'R' in state[1]:
            for comb in combinations:
                actions.append(self.pickups_combs(comb))          
        else:
            for comb in combinations:
                actions.append(comb+tuple((False)*self.V)) # N represents not doing anything
                
        return actions
    
    def goal_test(self, state):
        """Return True if the state is a goal"""
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
                        
                for r in self.requests:
                    self.pickups.append(r[2])  
                          
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
                    vehicule_status.append([v,0,[-1]*self.lugares[v]]) # [numero do veiculo,posição atual do veiculo,[0,0,0,0] cada numero representa o nº do request da pessoa]
                    # (-1) representa lugar vago         
        search.Problem.__init__(self, initial=[request_status, vehicule_status, 0] ) #path_cost e parent são feitos automaticamente! ultimo elemento é o tempo
        
    
    def solve(self):
        """Calls the uninformed search algorithm chosen. Returns solutions in the specified format"""
        search.uniform_cost_search(FleetProblem())
        # tem de percorrer a solução e fazer print da solução no formato do 1ºassignment!!!!
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
