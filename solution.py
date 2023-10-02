import search
import numpy as np
import io
import itertools


class FleetProblem (search.Problem) :
    
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
    
    def __init__(self,fh):
        self.load(fh)
        pass
        
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
    
    def update_drop_Off (self, state):
        """Updates request status from pickupp to dropoff and vehicle capacity. Vehicle final positions have to be already updated! """
        temp_state = state
            
        if 'P' in temp_state[1]:
            for i in enumerate(temp_state[1]):
                if temp_state[1][i] == 'P':
                    car_numb=0
                    seat_numb=0
                    
                    #find car thats responsible for said request
                    for car in enumerate(temp_state):
                        if car == 0:
                            continue
                        for seat in enumerate(temp_state[car][2]):
                            if temp_state[car][2][seat] == i: #if the car has people from request i
                                car_numb = car
                                seat_numb = seat
                                break
                    temp_state[car_numb][2][seat_numb] = -1 # changing request to none inside car
                    temp_state[car_numb][-1] += self.requests[i][-1] # freeing up seats
                    temp_state[1][i] = 'D' # updating request status
                    
        return temp_state
    
    def result (self, state, action):
        """Return the state that results from exectuting given action in given state"""
        
        initial_positions = []
        temp_state = state
        for v in self.V:
            initial_positions.append(state[v+1][1]) # get initial positions of vehicles
            
        for v in self.V:
            temp_state[v+1][1] = action[v][1] # update positions of vehicles
                
        temp_state[-1] += self.find_max(initial_positions,action) # update the time 
            
        temp_state = self.check_Request_ready(temp_state) # update from Not Ready to Ready
        
        for pos in enumerate(action):
            for a in enumerate(action[pos]):
                if  a!=0:
                    if pos[a]: # its true if the people were picked up
                        temp_state[1][a-1] = 'P' #update status from pickup 
                        people = self.requests[a-1][-1]
                        temp_state[pos+1][-1] -= people # updating number of free seats in car
                        
                        for i in range(self.lugares[pos]):# updating list of which people are inside the car
                            if temp_state[pos+1][2][i] == -1 :
                                temp_state[pos+1][2][i] = a-1
                                break
        #update for dropoffs
        temp_state = self.update_drop_Off(temp_state)
        
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
            if self.action_val(action) == False:
                actions.remove(action)
        '''
        action format:
        [final position,[T,F,list saying which requests to pickup or not]] * numero de veiculos
        para cada ação!
        '''
        return actions
    
    def goal_test(self, state):
        """Return True if the state is a goal"""
        
        goal = ['D']*self.R
        
        if state[1] == goal:
            return True
        else:
            return False
    
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
        
    
    def solve(self,fh):
        """Calls the uninformed search algorithm chosen. Returns solutions in the specified format"""
        search.uniform_cost_search(FleetProblem(fh=fh),display=True)
        # tem de percorrer a solução e fazer print da solução no formato do 1ºassignment!!!!---------------------------unfinished--------------------
        pass
        
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
        problem.solve(fh)#--------------------------------------problem calling problem nees to init_ and need to pass fh

        
if __name__=='__main__':
    main()
