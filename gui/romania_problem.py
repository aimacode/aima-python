from tkinter import *
import sys
import os.path
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search import *

root = None
city_coord = {}
romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
algo=None
start=None
goal=None


def create_map(root):
    '''
    This function draws out the required map.
    '''
    global romania_problem
    romania_locations = romania_map.locations
    width = 750
    height = 670
    margin = 5
    city_map = Canvas(root, width=width, height=height)
    city_map.pack()

    # Since lines have to be drawn between particular points, we need to list
    # them separately
    make_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        romania_map.get('Arad', 'Sibiu'))
    make_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        romania_map.get('Arad', 'Zerind'))
    make_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        romania_map.get('Arad', 'Timisoara'))
    make_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        romania_map.get('Oradea', 'Zerind'))
    make_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        romania_map.get('Oradea', 'Sibiu'))
    make_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        romania_map.get('Lugoj', 'Timisoara'))
    make_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        romania_map.get('Lugoj', 'Mehandia'))
    make_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        romania_map.get('Drobeta', 'Mehandia'))
    make_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        romania_map.get('Drobeta', 'Craiova'))
    make_line(
        city_map,
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        romania_map.get('Pitesti', 'Craiova'))
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        romania_map.get('Rimnicu', 'Craiova'))
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        romania_map.get('Rimnicu', 'Sibiu'))
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        romania_map.get('Rimnicu', 'Pitesti'))
    make_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        romania_map.get('Fagaras', 'Sibiu'))
    make_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        romania_map.get('Fagaras', 'Bucharest'))
    make_line(
        city_map,
        romania_locations['Giurgiu'][0],
        height -
        romania_locations['Giurgiu'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        romania_map.get('Giurgiu', 'Bucharest'))
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        romania_map.get('Urziceni', 'Bucharest'))
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        romania_map.get('Urziceni', 'Hirsova'))
    make_line(
        city_map,
        romania_locations['Eforie'][0],
        height -
        romania_locations['Eforie'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        romania_map.get('Eforie', 'Hirsova'))
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        romania_map.get('Urziceni', 'Vaslui'))
    make_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        romania_map.get('Iasi', 'Vaslui'))
    make_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Neamt'][0],
        height -
        romania_locations['Neamt'][1],
        romania_map.get('Iasi', 'Neamt'))

    for city in romania_locations.keys():
        make_rectangle(
            city_map,
            romania_locations[city][0],
            height -
            romania_locations[city][1],
            margin,
            city)

    make_legend(city_map)

def make_line(map, x0, y0, x1, y1, distance):
    '''
    This function draws out the lines joining various points.
    '''
    map.create_line(x0, y0, x1, y1)
    map.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=distance)


def make_rectangle(map, x0, y0, margin, city_name):
    '''
    This function draws out rectangles for various points.
    '''
    global city_coord
    rect = map.create_rectangle(
        x0 - margin,
        y0 - margin,
        x0 + margin,
        y0 + margin,
        fill="white")
    map.create_text(
        x0 - 2 * margin,
        y0 - 2 * margin,
        text=city_name,
        anchor=SE)
    city_coord.update({city_name: rect})

def make_legend(map):
    
    rect1=map.create_rectangle(600,100,610,110,fill="white") 
    text1 = map.create_text(615, 105,anchor=W,text="Un-explored")
    
    rect2=map.create_rectangle(600,115,610,125,fill="orange")
    text2 = map.create_text(615, 120,anchor=W,text="Frontier")
    
    rect3=map.create_rectangle(600,130,610,140,fill="red")
    text3 = map.create_text(615, 135,anchor=W,text="Currently Exploring")
    
    rect4=map.create_rectangle(600,145,610,155,fill="grey")
    text4 = map.create_text(615, 150,anchor=W,text="Explored")
    
    rect5=map.create_rectangle(600,160,610,170,fill="dark green")
    text5 = map.create_text(615, 165, anchor=W, text="Final Solution")
    
def main():
    global algo,start,goal
    root = Tk()
    root.title("Road Map of Romania")
    root.geometry("950x1150")
    algo=StringVar(root)
    start = StringVar(root)
    goal = StringVar(root)
    algo.set("Breadth First Tree Search")
    start.set('Arad')
    goal.set('Bucharest')
    cities=list(romania_map.locations.keys())
    cities.sort()
    algorithm_menu=OptionMenu(root,algo,"Breadth-First Tree Search","Depth-First Tree Search")
    Label(root,text="\n Search Algorithm").pack()
    algorithm_menu.pack()
    Label(root,text="\n Start City").pack()
    start_menu = OptionMenu(root,start,*cities)
    start_menu.pack()
    Label(root, text="\n Goal City").pack()
    goal_menu = OptionMenu(root,goal,*cities)
    goal_menu.pack()
    next_button = Button(root, width=6, height=2, text="Next", command=None,padx=2,pady=2,relief=GROOVE)
    next_button.pack(side=BOTTOM)
    create_map(root)
    root.mainloop()


if __name__ == "__main__":
    main()
