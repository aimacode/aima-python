from tkinter import *
import sys
import os.path
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from search import *
from search import breadth_first_tree_search as bfts, depth_first_tree_search as dfts
from utils import Stack, FIFOQueue, PriorityQueue
from copy import deepcopy
root = None
city_coord = {}
romania_problem = None
algo = None
start = None
goal = None
counter = -1
city_map = None
frontier = None
front = None
node = None
next_button = None


def create_map(root):
    '''
    This function draws out the required map.
    '''
    global city_map, start, goal
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

    rect1 = map.create_rectangle(600, 100, 610, 110, fill="white")
    text1 = map.create_text(615, 105, anchor=W, text="Un-explored")

    rect2 = map.create_rectangle(600, 115, 610, 125, fill="orange")
    text2 = map.create_text(615, 120, anchor=W, text="Frontier")

    rect3 = map.create_rectangle(600, 130, 610, 140, fill="red")
    text3 = map.create_text(615, 135, anchor=W, text="Currently Exploring")

    rect4 = map.create_rectangle(600, 145, 610, 155, fill="grey")
    text4 = map.create_text(615, 150, anchor=W, text="Explored")

    rect5 = map.create_rectangle(600, 160, 610, 170, fill="dark green")
    text5 = map.create_text(615, 165, anchor=W, text="Final Solution")


def tree_search(problem):
    '''
    This function has been changed to make it suitable for the Tkinter GUI.
    '''
    global counter, frontier, node
    # print(counter)
    if counter == -1:
        frontier.append(Node(problem.initial))
        # print(frontier)
        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()
        # print(node)
        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            # print(node)
            return node
        frontier.extend(node.expand(problem))
        # print(frontier)
        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        # print(node)
        display_explored(node)
    return None


def display_frontier(queue):
    '''
    This function marks the frontier nodes (orange) on the map.
    '''
    global city_map, city_coord
    qu = deepcopy(queue)
    while qu:
        node = qu.pop()
        for city in city_coord.keys():
            if node.state == city:
                city_map.itemconfig(city_coord[city], fill="orange")
    return


def display_current(node):
    '''
    This function marks the currently exploring node (red) on the map.
    '''
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="red")
    return


def display_explored(node):
    '''
    This function marks the already explored node (gray) on the map.
    '''
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="gray")
    return


def display_final(cities):
    '''
    This function marks the final solution nodes (green) on the map.
    '''
    global city_map, city_coord
    for city in cities:
        city_map.itemconfig(city_coord[city], fill="green")
    return


def breadth_first_tree_search(problem):
    """Search the shallowest nodes in the search tree first."""
    global frontier, counter
    if counter == -1:
        frontier = FIFOQueue()
    return tree_search(problem)


def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first."""
    global frontier,counter
    if counter == -1:
        frontier=Stack()
    return tree_search(problem)

# TODO: Remove redundant code.
def on_click():
    '''
    This function defines the action of the 'Next' button.
    '''
    global algo, counter, next_button, romania_problem, start, goal
    romania_problem = GraphProblem(start.get(), goal.get(), romania_map)
    if "Breadth-First Tree Search" == algo.get():
        node = breadth_first_tree_search(romania_problem)
        if node is not None:
            final_path = bfts(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "Depth-First Tree Search" == algo.get():
        node = depth_first_tree_search(romania_problem)
        if node is not None:
            final_path = dfts(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1

def reset_map():
    global counter, city_coord, city_map, next_button
    counter = -1
    for city in city_coord.keys():
        city_map.itemconfig(city_coord[city], fill="white")
    next_button.config(state="normal")

# TODO: Add more search algorithms in the OptionMenu


def main():
    global algo, start, goal, next_button
    root = Tk()
    root.title("Road Map of Romania")
    root.geometry("950x1150")
    algo = StringVar(root)
    start = StringVar(root)
    goal = StringVar(root)
    algo.set("Breadth-First Tree Search")
    start.set('Arad')
    goal.set('Bucharest')
    cities = sorted(romania_map.locations.keys())
    algorithm_menu = OptionMenu(
        root, algo, "Breadth-First Tree Search", "Depth-First Tree Search")
    Label(root, text="\n Search Algorithm").pack()
    algorithm_menu.pack()
    Label(root, text="\n Start City").pack()
    start_menu = OptionMenu(root, start, *cities)
    start_menu.pack()
    Label(root, text="\n Goal City").pack()
    goal_menu = OptionMenu(root, goal, *cities)
    goal_menu.pack()
    frame1 = Frame(root)
    next_button = Button(
        frame1,
        width=6,
        height=2,
        text="Next",
        command=on_click,
        padx=2,
        pady=2,
        relief=GROOVE)
    next_button.pack(side=RIGHT)
    reset_button = Button(
        frame1,
        width=6,
        height=2,
        text="Reset",
        command=reset_map,
        padx=2,
        pady=2,
        relief=GROOVE)
    reset_button.pack(side=RIGHT)
    frame1.pack(side=BOTTOM)
    create_map(root)
    root.mainloop()


if __name__ == "__main__":
    main()
