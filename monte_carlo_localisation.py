"""
    For simplicity the positive orientation of theta is reversed
    i.e - when moving clockwise theta is positive
"""


from math import *
import matplotlib.pyplot as plt
import random
from robot_class import robot

world_size = 9.0

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
#this command is given to keep the readability in of the maze while making computation simpler
maze = list(zip(*maze))


def eval(r, p):
    sum = 0.0;
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err
    return sum / float(len(p))



####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####
myrobot = robot()
myrobot.set(1.0, 6.0, 0)
"""
myrobot.set(6.0, 3.0, pi/4)
myrobot.set_sense_orient()
print myrobot.sense()
"""
#print myrobot.convert_to_degrees(myrobot.set_sense_orient())
#print myrobot.set_sense_orient()
#print myrobot.x, myrobot.y
"""
x =  0.5
#print -3.5 % 2
#print myrobot.orientation * 180 / pi_c
print (x * 2 * pi) % (2*pi)
"""

n = 1000
t = 100

p = []
"""
xs = []
ys = []
for i in range(len(maze)):
    for j in range(len(maze[0])):
        if maze[i][j] == 1:
            xs.append(i)
            ys.append(j)
plt.plot(xs,ys,'--')
"""

for i in range(n):
    x = robot()
    x.set_noise(0.1, 0.01, 2.0)
    p.append(x)
    plt.plot(x.x, x.y , ".")
#plt.axis([0,10,0,10])
plt.show()
for i in range(t):
    myrobot = myrobot.move(0, 0.5)
    myrobot.set_sense_orient()
    z = myrobot.sense()

    p2 = []
    for i in range(n):
        p2.append(p[i].move(0, 0.5))
    p = p2

    w = []
    for i in range(n):
        w.append(p[i].measurement_prob(z))
    #print w

    p3 = []
    index = int(random.random() * n)
    beta = 0
    mw = max(w)
    for i in range(n):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % n
        p3.append(p[index])
    p = p3
    #print p
    print(eval(myrobot, p))
    for i in range(n):
    	plt.plot(p[i].x,p[i].y, '.')
    plt.axis([0,10,0,10])
    plt.show()