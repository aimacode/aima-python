class SearchNode:
    def __init__(self, data, parent, children=list()):
        self.data = data
        self.parent = parent
        self.children = children


def horizontalDistance(node):
    if node.children == []:
            return 2
    total = 0
    for child in node.children:        
        total += horizontalDistance(child)

    return len(node.children) - 1 + total


def main():
    root = SearchNode("a",None)
    parent1 = SearchNode("parent1", root)
    child1 = SearchNode("child1", parent1)
    child2 = SearchNode("child2", parent1)
    child3 = SearchNode("child3", parent1)
    child4 = SearchNode("child4", parent1)

    parent2 = SearchNode("parent2", root)
    child5 = SearchNode("child5", parent2)
    child6 = SearchNode("child6", parent2)
    child7 = SearchNode("child7", parent2)

    root.children = [parent1, parent2]
    parent1.children = [child1, child2, child3, child4]
    parent2.children = [child5, child6, child7]

    print(horizontalDistance(root))

if __name__ == "__main__":
    main()