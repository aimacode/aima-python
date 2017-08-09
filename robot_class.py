import random
from math import *

maze = ( ( 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 ),
         ( 1, 1, 0, 0, 1, 1, 0, 0, 0, 0 ),
         ( 0, 1, 1, 0, 0, 0, 0, 1, 0, 1 ),
         ( 0, 0, 0, 0, 1, 0, 0, 1, 1, 1 ),
         ( 1, 1, 0, 1, 1, 1, 0, 0, 1, 0 ),
         ( 1, 1, 1, 0, 1, 1, 1, 0, 1, 0 ),
         ( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
         ( 1, 1, 0, 1, 1, 1, 1, 0, 0, 0 ),
         ( 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ),
         ( 0, 0, 1, 0, 0, 1, 1, 1, 1, 0 ))
maze = list(zip(*maze))

class robot:
    def __init__(self):
        self.world_size = 9.0
        self.x = random.random() * self.world_size
        self.y = random.random() * self.world_size
        self.orientation = random.random() * 2.0 * pi
        self.sense_orient = [0,pi,pi/2,-pi/2]
        self.forward_noise = 0.0
        self.turn_noise    = 0.0
        self.sense_noise   = 0.0
    
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= self.world_size:
            raise ValueError('X coordinate out of bound')
        if new_y < 0 or new_y >= self.world_size:
            raise ValueError('Y coordinate out of bound')
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError('Orientation must be in [0..2pi]')
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    
    def sense(self):
        sensor_reading = [0,0,0,0]
        x2 = float("Inf")
        y2 = float("Inf")
        #self.x = 2.0
        #self.y = 3.0
        #self.sense_orient = [7*pi/4, pi, pi/2, 3*pi/2]
        #print self.sense_orient
        for i in range(len(self.sense_orient)):
            if (self.sense_orient[i] > 0 and self.sense_orient[i] < pi/2):
                #taking ceil because the nearest wall in the direction of sensor is needed.
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, int(self.world_size)):
                    x_at_y = (j - y_intercept)/slope
                    #taking floor because it makes the calculations easy
                    if x_at_y > x:
                        #go right
                        x += 1
                        if x >= self.world_size - 1:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y < x:
                        #go down
                        y += 1
                        if y >= self.world_size - 1:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x += 1
                        y += 1
                        if x >= self.world_size - 1 or y >= self.world_size:
                            break
                        # this will change for a real robot
                        if (maze[x - 1][y] == 1 and maze[x][y - 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break

            elif (self.sense_orient[i] > pi/2 and self.sense_orient[i] < pi):
                x = int(floor(self.x))
                y = int(ceil(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, int(self.world_size)):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y < x:
                        #go left
                        x -= 1
                        if x < 0:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y > x:
                        #go down
                        y += 1
                        if y >= self.world_size - 1:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x -= 1
                        y += 1
                        if x < 0 or y >= self.world_size - 1:
                            break
                        # this will change for a real robot
                        if (maze[x + 1][y] == 1 and maze[x][y - 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] > pi and self.sense_orient[i] < 3*pi/2):
                x = int(floor(self.x))
                y = int(floor(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, 0, -1):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y < x:
                        #go left
                        x -= 1
                        if x < 0:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y > x:
                        #go up
                        y -= 1
                        if y < 0:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x -= 1
                        y -= 1
                        if x < 0 or y < 0:
                            break
                        # this will change for a real robot
                        if (maze[x + 1][y] == 1 and maze[x][y + 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] > 3*pi/2 and self.sense_orient[i] < 2*pi):
                x = int(ceil(self.x))
                y = int(floor(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, 0, -1):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y > x:
                        #go right
                        x += 1
                        if x >= self.world_size - 1:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y < x:
                        #go up
                        y -= 1
                        if y < 0:
                            break
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x += 1
                        y -= 1
                        if x >= self.world_size - 1 or y < 0:
                            break
                        # this will change for a real robot
                        if (maze[x - 1][y] == 1 and maze[x][y + 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] == 0):
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                y2 = self.y
                for j in range(x, int(self.world_size)):
                    if maze[j][y] == 1:
                        x2 = j
                        break
                else:
                    x2 = self.world_size
            elif (self.sense_orient[i] == pi/2):
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                x2 = self.x
                #there might be error due to negative
                for j in range(y, int(self.world_size)):
                    if maze[x][j] == 1:
                        y2 = j
                        break
                else:
                    y2 = 0
            elif (self.sense_orient[i] == pi):
                x = int(floor(self.x))
                y = int(ceil(self.y))
                y2 = self.y
                #there might be error due to negative
                for j in range(x, 0, -1):
                    if maze[j][y] == 1:
                        x2 = j
                        break
                else:
                    x2 = 0
            elif (self.sense_orient[i] == 3*pi/2):
                x = int(ceil(self.x))
                y = int(floor(self.y))
                x2 = self.x
                #there might be error due to negative
                for j in range(y, 0, -1):
                    if maze[x][j] == 1:
                        y2 = j
                        break
                else:
                    y2 = 0
            else:
                print("how")
            #print x2, y2
            sensor_reading[i] = sqrt((y2 - self.y) ** 2 + (x2 - self.x) ** 2)
            #print sensor_reading[i]
            sensor_reading[i] += random.gauss(0.0, self.sense_noise)
        return sensor_reading
    
    def set_sense_orient(self):
        orientation = self.orientation
        #set forward
        self.sense_orient[0] = orientation
        #set reverse
        self.sense_orient[1] = (orientation + pi) % (2*pi)
        #set right
        self.sense_orient[2] = (orientation + (pi/2)) % (2*pi)
        #set left
        self.sense_orient[3] = (orientation - (pi/2)) % (2*pi)

        return self.sense_orient
    
    def convert_to_degrees(self, array):
        for i in range(len(array)):
            array[i] = array[i] * 180 / pi
        return array

    def move(self, turn, forward):
        if forward < 0:
            raise ValueError('Robot cant move backwards')         
        
        # turn, and add randomness to the turning command
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi
        
        # move, and add randomness to the motion command
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= self.world_size    # cyclic truncate
        y %= self.world_size
        
        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        
        # calculates the probability of x for 1-dim Gaussian with mean mu and var. sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))
    
    
    def measurement_prob(self, measurement):
        
        # calculates how likely a measurement should be
        
        prob = 1.0;
        for i in range(len(self.sense_orient)):
            if (self.sense_orient[i] > 0 and self.sense_orient[i] < pi/2):
                #taking ceil because the nearest wall in the direction of sensor is needed.
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, int(self.world_size)):
                    x_at_y = (j - y_intercept)/slope
                    #taking floor because it makes the calculations easy
                    if x_at_y > x:
                        #go right
                        x += 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y < x:
                        #go down
                        y += 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x += 1
                        y += 1
                        # this will change for a real robot
                        if (maze[x - 1][y] == 1 and maze[x][y - 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break

            elif (self.sense_orient[i] > pi/2 and self.sense_orient[i] < pi):
                x = int(floor(self.x))
                y = int(ceil(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, int(self.world_size)):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y < x:
                        #go left
                        x -= 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y > x:
                        #go down
                        y += 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x -= 1
                        y += 1
                        # this will change for a real robot
                        if (maze[x + 1][y] == 1 and maze[x][y - 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] > pi and self.sense_orient[i] < 3*pi/2):
                x = int(floor(self.x))
                y = int(floor(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, 0, -1):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y < x:
                        #go left
                        x -= 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y > x:
                        #go up
                        y -= 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x -= 1
                        y -= 1
                        # this will change for a real robot
                        if (maze[x + 1][y] == 1 and maze[x][y + 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] > 3*pi/2 and self.sense_orient[i] < 2*pi):
                x = int(ceil(self.x))
                y = int(floor(self.y))
                theta = self.sense_orient[i]
                slope = tan(theta)
                y_intercept = self.y - slope * self.x

                for j in range(y, 0, -1):
                    x_at_y = (j - y_intercept)/slope
                    #taking ceil and floor different for both to make calculations easy
                    if x_at_y > x:
                        #go right
                        x += 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    elif x_at_y < x:
                        #go up
                        y -= 1
                        if maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
                    else:
                        x += 1
                        y -= 1
                        # this will change for a real robot
                        if (maze[x - 1][y] == 1 and maze[x][y + 1] == 1) or maze[x][y] == 1:
                            y2 = y
                            x2 = x
                            break
            elif (self.sense_orient[i] == 0):
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                y2 = self.y
                for j in range(x, int(self.world_size)):
                    if maze[j][y] == 1:
                        x2 = j
                        break
                else:
                    x2 = self.world_size
            elif (self.sense_orient[i] == pi/2):
                x = int(ceil(self.x))
                y = int(ceil(self.y))
                x2 = self.x
                #there might be error due to negative
                for j in range(y, int(self.world_size)):
                    if maze[x][j] == 1:
                        y2 = j
                        break
                else:
                    y2 = 0
            elif (self.sense_orient[i] == pi):
                x = int(floor(self.x))
                y = int(ceil(self.y))
                y2 = self.y
                #there might be error due to negative
                for j in range(x, 0, -1):
                    if maze[j][y] == 1:
                        x2 = j
                        break
                else:
                    x2 = 0
            elif (self.sense_orient[i] == 3*pi/2):
                x = int(ceil(self.x))
                y = int(floor(self.y))
                x2 = self.x
                #there might be error due to negative
                for j in range(y, 0, -1):
                    if maze[x][j] == 1:
                        y2 = j
                        break
                else:
                    y2 = 0
            #print x2, y2
            dist = sqrt((y2 - self.y) ** 2 + (x2 - self.x) ** 2)

            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])
        return prob
    
    
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))