from tkinter import *
import sys
import os.path
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search import *

root = None
city_coord = {}
height = 0


def create_map(root):
    '''
    This function draws out the required map.
    '''
    romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
    romania_locations = romania_map.locations
    global height
    width = 700
    height = 800
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
        140)
    make_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        75)
    make_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        118)
    make_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        71)
    make_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        151)
    make_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        111)
    make_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        70)
    make_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        75)
    make_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        120)
    make_line(
        city_map,
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        138)
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        146)
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        80)
    make_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        97)
    make_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        99)
    make_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        211)
    make_line(
        city_map,
        romania_locations['Giurgiu'][0],
        height -
        romania_locations['Giurgiu'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        90)
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        85)
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        98)
    make_line(
        city_map,
        romania_locations['Eforie'][0],
        height -
        romania_locations['Eforie'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        86)
    make_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        142)
    make_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        92)
    make_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Neamt'][0],
        height -
        romania_locations['Neamt'][1],
        87)

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
    root = Tk()
    root.title("Road Map of Romania")
    create_map(root)
    root.mainloop()


if __name__ == "__main__":
    main()
