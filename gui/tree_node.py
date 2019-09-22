from tkinter import Tk, Canvas
from math import floor


class TreeNode:
    def __init__(self, data, parent, children=list()):
        self.data = data
        self.parent = parent
        self.children = children
        self.status = 'f'
        self.width = 2
        self.x = None
        self.y = None

    def draw_dots(self, canvas):
        if self.x is None and self.y is None:
            self.x = self.width / 2.0 + 10.0
            self.y = 5.0

        canvas.create_oval(self.x * 20 + 20, self.y * 30 + 20, self.x * 20 - 20, self.y * 30 - 20, fill='orange')
        canvas.create_text(self.x * 20, self.y * 30, text=self.data[0])
        if self.parent is not None:
            canvas.create_line(self.parent.x * 20, self.parent.y * 30 + 20, self.x * 20, self.y * 30 - 20) 

        if self.children:
            middle_idx = len(self.children) // 2

            if len(self.children) % 2 != 0:
                middle_child = self.children[middle_idx]
                middle_child.x = self.x
                middle_child.y = self.y + 2
                middle_child.draw_dots(canvas)

                for idx in range(0, middle_idx):
                    self.draw_odd_child(idx, middle_idx, canvas)

                for idx in range(middle_idx + 1, len(self.children)):
                    self.draw_odd_child(idx, middle_idx, canvas)

            else:
                for idx in range(0, len(self.children)):
                    self.draw_even_child(idx, middle_idx, canvas)

    def draw_odd_child(self, idx, middle_idx, canvas):
        child = self.children[idx]
        padding = 0
        if idx < middle_idx:
            for i in range(idx + 1, middle_idx):
                padding += self.children[i].width
            padding += floor(len(self.children) // 2)
            padding += self.children[middle_idx].width / 2.0
            padding += child.width / 2.0
            child.x = self.x - padding
            child.y = self.y + 2
        elif idx > middle_idx:
            for i in range(middle_idx + 1, idx):
                padding += self.children[i].width
            padding += floor(len(self.children) // 2)
            padding += self.children[middle_idx].width / 2.0
            padding += child.width / 2.0
            child.x = self.x + padding
            child.y = self.y + 2
        child.draw_dots(canvas)

    def draw_even_child(self, idx, middle_idx, canvas):
        child = self.children[idx]
        padding = 0
        if idx < middle_idx:
            for i in range(idx + 1, middle_idx):
                padding += self.children[i].width
            padding += (len(self.children) // 2) - idx - 1
            padding += 0.5
            padding += child.width / 2
            child.x = self.x - padding
            child.y = self.y + 2
        else:
            for i in range(middle_idx, idx):
                padding += self.children[i].width
            padding += idx - (len(self.children) // 2)
            padding += 0.5
            padding += child.width / 2
            child.x = self.x + padding
            child.y = self.y + 2
        child.draw_dots(canvas)

    def horizontal_distance(self):
        if not self.children:
            return 2

        total = 0
        for child in self.children:
            total += child.horizontal_distance()

        self.width = total + len(self.children) - 1
        return self.width


def main():
    root = TreeNode("root", None)
    parent1 = TreeNode("parent1", root)
    child1 = TreeNode("child1", parent1)
    child2 = TreeNode("child2", parent1)
    child3 = TreeNode("child3", parent1)
    child4 = TreeNode("child4", parent1)

    parent2 = TreeNode("parent2", root)
    child5 = TreeNode("child5", parent2)
    child6 = TreeNode("child6", parent2)
    child7 = TreeNode("child7", parent2)
    child8 = TreeNode('child8', parent2)
    # child10 = SearchNode('child10', parent1)

    grandchild1 = TreeNode("grandchild1", child6)
    grandchild2 = TreeNode("grandchild2", child7)
    # child9 = SearchNode('child9', parent3)
    root.children = [parent1, parent2]
    parent1.children = [child1, child2, child3, child4]
    parent2.children = [child5, child6, child7, child8]
    child6.children = [grandchild1]
    child7.children = [grandchild2]

    print(root.horizontal_distance())

    window = Tk()
    window.geometry('950x1100')
    canvas = Canvas(window, width=950, height=1100)
    canvas.pack()

    root.draw_dots(canvas)
    window.mainloop()


if __name__ == "__main__":
    main()
