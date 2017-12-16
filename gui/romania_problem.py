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

    for city in romania_locations.keys():
        create_rectangle(
            city_map,
            romania_locations[city][0],
            height -
            romania_locations[city][1],
            margin,
            city)

    # Since lines have to be drawn between particular points, we need to list
    # them separately
    create_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        140)
    create_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        75)
    create_line(
        city_map,
        romania_locations['Arad'][0],
        height -
        romania_locations['Arad'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        118)
    create_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Zerind'][0],
        height -
        romania_locations['Zerind'][1],
        71)
    create_line(
        city_map,
        romania_locations['Oradea'][0],
        height -
        romania_locations['Oradea'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        151)
    create_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Timisoara'][0],
        height -
        romania_locations['Timisoara'][1],
        111)
    create_line(
        city_map,
        romania_locations['Lugoj'][0],
        height -
        romania_locations['Lugoj'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        70)
    create_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        75)
    create_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        120)
    create_line(
        city_map,
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        138)
    create_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Craiova'][0],
        height -
        romania_locations['Craiova'][1],
        146)
    create_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        80)
    create_line(
        city_map,
        romania_locations['Rimnicu'][0],
        height -
        romania_locations['Rimnicu'][1],
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        97)
    create_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Sibiu'][0],
        height -
        romania_locations['Sibiu'][1],
        99)
    create_line(
        city_map,
        romania_locations['Fagaras'][0],
        height -
        romania_locations['Fagaras'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        211)
    create_line(
        city_map,
        romania_locations['Giurgiu'][0],
        height -
        romania_locations['Giurgiu'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        90)
    create_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        85)
    create_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        98)
    create_line(
        city_map,
        romania_locations['Eforie'][0],
        height -
        romania_locations['Eforie'][1],
        romania_locations['Hirsova'][0],
        height -
        romania_locations['Hirsova'][1],
        86)
    create_line(
        city_map,
        romania_locations['Urziceni'][0],
        height -
        romania_locations['Urziceni'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        142)
    create_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Vaslui'][0],
        height -
        romania_locations['Vaslui'][1],
        92)
    create_line(
        city_map,
        romania_locations['Iasi'][0],
        height -
        romania_locations['Iasi'][1],
        romania_locations['Neamt'][0],
        height -
        romania_locations['Neamt'][1],
        87)


def create_line(map, x0, y0, x1, y1, distance):
    '''
    This function draws out the lines joining various points.
    '''
    map.create_line(x0, y0, x1, y1)
    map.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=distance)


def create_rectangle(map, x0, y0, margin, city_name):
    '''
    This function draws out rectangles for various points.
    '''
    global city_coord
    rect = map.create_rectangle(
        x0 - margin,
        y0 - margin,
        x0 + margin,
        y0 + margin)
    map.create_text(
        x0 - 2 * margin,
        y0 - 2 * margin,
        text=city_name,
        anchor=SE)
    city_coord.update({city_name: rect})


def main():
    root = Tk()
    root.title("Road Map of Romania")
    create_map(root)
    root.mainloop()


if __name__ == "__main__":
    main()
