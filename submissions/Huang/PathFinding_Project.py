import math

class Node():

    def __init__(self, value, position):
        self.value = value
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.m = 0

    def __eq__(self, other):
        return self.position == other.position

def currentMaze(maze, node_position, temp_passed_list):
    collected = False;
    if maze[node_position[0]][node_position[1]] < 1:
        return True
    for open_node in temp_passed_list:
        if maze[open_node.position[0]][open_node.position[1]] == -1:
            collected = True
    if (collected == True) and (maze[node_position[0]][node_position[1]] == 1):
        return True
    return False

def astar(maze, start, end):

    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    start_node.m = maze[start_node.position[0]][start_node.position[1]]
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0
    end_node.m = maze[end_node.position[0]][end_node.position[1]]
    for x in range((len(maze) - 1)):
        for y in range((len(maze[len(maze) - 1]) - 1)):
            current_node = Node(None, [x,y])
            current_node.m = maze[current_node.position[0]][current_node.position[1]]

    open_list = []
    closed_list = []
    temp_passed_list = []
    shortcut = 0
    sword = Node(None, None)

    open_list.append(start_node)

    while len(open_list) > 0:

        collected = False
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)
        temp_passed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.value
            return path[::-1]

        children = []

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            #if maze[node_position[0]][node_position[1]] == 2 :
            #    continue

            #if (collected == False) and (current_node.m == 1):
            #    continue

            if maze[current_node.position[0]][current_node.position[1]] == -1:
                collected = True
                sword = Node(None, (current_node.position[0], current_node.position[1]))

            if (maze[current_node.position[0]][current_node.position[1]] == 1) and (collected == True):
                shortcut  = math.sqrt(((start_node.position[0] - end_node.position[0]) ** 2) + ((start.position[1] - end_node.position[1]) ** 2)) - math.sqrt(((sword.position[0] - end_node.position[0]) ** 2) + ((sword.position[1] - end_node.position[1]) ** 2))

            if currentMaze(maze, node_position, temp_passed_list) == False:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:

            #for closed_child in closed_list:
            #    if child == closed_child:
            #        continue

            child.g = current_node.g + 1
            child.h = math.sqrt(((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)) - shortcut
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)



def main():

    maze = [[0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    start = (0, 0)
    end = (6, 8)

    path = astar(maze, start, end)
    print(path)


if __name__ == '__main__':
    main()