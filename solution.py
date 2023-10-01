import search


class FleetProblem( search.Problem ) :
    
    R=0
    P=0
    V=0
    lugares=0
    requests=0
    m_Time=0
    
    def result (self, state, action):
        """Return the state that results from exectuting given action in given state"""
        pass
    def actions (self, state):
        """Return actions that can be executed in the given state"""
        pass
    def goaltest(self, state):
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
                        
                continue
            
            elif line[0] == 'V':
                
                newline = line[1:]
                self.V = int(newline)
                self.lugares = [0]*self.V
                
                for x in range(self.V):
                    
                    line = fh.readline()
                    self.lugares[x] = int(line)
                    
        self.initial 
        pass
    def solve(self):
        """Calls the uninformed search algorithm chosen. Returns solutions in the specified format"""
        search.
        pass
