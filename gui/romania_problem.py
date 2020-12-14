from copy import deepcopy
from tkinter import *

from search import *
from utils import PriorityQueue

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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
explored = None


def create_map(root):
    """This function draws out the required map."""
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
        romania_map.get('Lugoj', 'Mehadia'))
    make_line(
        city_map,
        romania_locations['Drobeta'][0],
        height -
        romania_locations['Drobeta'][1],
        romania_locations['Mehadia'][0],
        height -
        romania_locations['Mehadia'][1],
        romania_map.get('Drobeta', 'Mehadia'))
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
        romania_locations['Bucharest'][0],
        height -
        romania_locations['Bucharest'][1],
        romania_locations['Pitesti'][0],
        height -
        romania_locations['Pitesti'][1],
        romania_map.get('Bucharest', 'Pitesti'))
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
    """This function draws out the lines joining various points."""
    map.create_line(x0, y0, x1, y1)
    map.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=distance)


def make_rectangle(map, x0, y0, margin, city_name):
    """This function draws out rectangles for various points."""
    global city_coord
    rect = map.create_rectangle(
        x0 - margin,
        y0 - margin,
        x0 + margin,
        y0 + margin,
        fill="white")
    if "Bucharest" in city_name or "Pitesti" in city_name or "Lugoj" in city_name \
            or "Mehadia" in city_name or "Drobeta" in city_name:
        map.create_text(
            x0 - 2 * margin,
            y0 - 2 * margin,
            text=city_name,
            anchor=E)
    else:
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
    """
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Don't worry about repeated paths to a state. [Figure 3.7]
    This function has been changed to make it suitable for the Tkinter GUI.
    """
    global counter, frontier, node

    if counter == -1:
        frontier.append(Node(problem.initial))

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def graph_search(problem):
    """
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    If two paths reach a state, only use the first one. [Figure 3.7]
    This function has been changed to make it suitable for the Tkinter GUI.
    """
    global counter, frontier, node, explored
    if counter == -1:
        frontier.append(Node(problem.initial))
        explored = set()

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def display_frontier(queue):
    """This function marks the frontier nodes (orange) on the map."""
    global city_map, city_coord
    qu = deepcopy(queue)
    while qu:
        node = qu.pop()
        for city in city_coord.keys():
            if node.state == city:
                city_map.itemconfig(city_coord[city], fill="orange")


def display_current(node):
    """This function marks the currently exploring node (red) on the map."""
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="red")


def display_explored(node):
    """This function marks the already explored node (gray) on the map."""
    global city_map, city_coord
    city = node.state
    city_map.itemconfig(city_coord[city], fill="gray")


def display_final(cities):
    """This function marks the final solution nodes (green) on the map."""
    global city_map, city_coord
    for city in cities:
        city_map.itemconfig(city_coord[city], fill="green")


def breadth_first_tree_search(problem):
    """Search the shallowest nodes in the search tree first."""
    global frontier, counter, node
    if counter == -1:
        frontier = deque()

    if counter == -1:
        frontier.append(Node(problem.initial))

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.popleft()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def depth_first_tree_search(problem):
    """Search the deepest nodes in the search tree first."""
    # This search algorithm might not work in case of repeated paths.
    global frontier, counter, node
    if counter == -1:
        frontier = []  # stack

    if counter == -1:
        frontier.append(Node(problem.initial))

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def breadth_first_graph_search(problem):
    """[Figure 3.11]"""
    global frontier, node, explored, counter
    if counter == -1:
        node = Node(problem.initial)
        display_current(node)
        if problem.goal_test(node.state):
            return node

        frontier = deque([node])  # FIFO queue

        display_frontier(frontier)
        explored = set()
    if counter % 3 == 0 and counter >= 0:
        node = frontier.popleft()
        display_current(node)
        explored.add(node.state)
    if counter % 3 == 1 and counter >= 0:
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def depth_first_graph_search(problem):
    """Search the deepest nodes in the search tree first."""
    global counter, frontier, node, explored
    if counter == -1:
        frontier = []  # stack
    if counter == -1:
        frontier.append(Node(problem.initial))
        explored = set()

        display_frontier(frontier)
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()

        display_current(node)
    if counter % 3 == 1 and counter >= 0:
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)

        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    global frontier, node, explored, counter

    if counter == -1:
        f = memoize(f, 'f')
        node = Node(problem.initial)
        display_current(node)
        if problem.goal_test(node.state):
            return node
        frontier = PriorityQueue('min', f)
        frontier.append(node)
        display_frontier(frontier)
        explored = set()
    if counter % 3 == 0 and counter >= 0:
        node = frontier.pop()
        display_current(node)
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
    if counter % 3 == 1 and counter >= 0:
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
        display_frontier(frontier)
    if counter % 3 == 2 and counter >= 0:
        display_explored(node)
    return None


def uniform_cost_search(problem):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost)


def astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))


# TODO:
# Remove redundant code.
# Make the interchangeability work between various algorithms at each step.
def on_click():
    """
    This function defines the action of the 'Next' button.
    """
    global algo, counter, next_button, romania_problem, start, goal
    romania_problem = GraphProblem(start.get(), goal.get(), romania_map)
    if "Breadth-First Tree Search" == algo.get():
        node = breadth_first_tree_search(romania_problem)
        if node is not None:
            final_path = breadth_first_tree_search(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "Depth-First Tree Search" == algo.get():
        node = depth_first_tree_search(romania_problem)
        if node is not None:
            final_path = depth_first_tree_search(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "Breadth-First Graph Search" == algo.get():
        node = breadth_first_graph_search(romania_problem)
        if node is not None:
            final_path = breadth_first_graph_search(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "Depth-First Graph Search" == algo.get():
        node = depth_first_graph_search(romania_problem)
        if node is not None:
            final_path = depth_first_graph_search(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "Uniform Cost Search" == algo.get():
        node = uniform_cost_search(romania_problem)
        if node is not None:
            final_path = uniform_cost_search(romania_problem).solution()
            final_path.append(start.get())
            display_final(final_path)
            next_button.config(state="disabled")
        counter += 1
    elif "A* - Search" == algo.get():
        node = astar_search(romania_problem)
        if node is not None:
            final_path = astar_search(romania_problem).solution()
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
if __name__ == "__main__":
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
        root,
        algo, "Breadth-First Tree Search", "Depth-First Tree Search",
        "Breadth-First Graph Search", "Depth-First Graph Search",
        "Uniform Cost Search", "A* - Search")
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
