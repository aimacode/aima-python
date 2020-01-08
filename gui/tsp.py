from tkinter import *
from tkinter import messagebox

import utils
from search import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

distances = {}


class TSProblem(Problem):
    """subclass of Problem to define various functions"""

    def two_opt(self, state):
        """Neighbour generating function for Traveling Salesman Problem"""
        neighbour_state = state[:]
        left = random.randint(0, len(neighbour_state) - 1)
        right = random.randint(0, len(neighbour_state) - 1)
        if left > right:
            left, right = right, left
        neighbour_state[left: right + 1] = reversed(neighbour_state[left: right + 1])
        return neighbour_state

    def actions(self, state):
        """action that can be executed in given state"""
        return [self.two_opt]

    def result(self, state, action):
        """result after applying the given action on the given state"""
        return action(state)

    def path_cost(self, c, state1, action, state2):
        """total distance for the Traveling Salesman to be covered if in state2"""
        cost = 0
        for i in range(len(state2) - 1):
            cost += distances[state2[i]][state2[i + 1]]
        cost += distances[state2[0]][state2[-1]]
        return cost

    def value(self, state):
        """value of path cost given negative for the given state"""
        return -1 * self.path_cost(None, None, None, state)


class TSPGui():
    """Class to create gui of Traveling Salesman using simulated annealing where one can
    select cities, change speed and temperature. Distances between cities are euclidean
    distances between them.
    """

    def __init__(self, root, all_cities):
        self.root = root
        self.vars = []
        self.frame_locations = {}
        self.calculate_canvas_size()
        self.button_text = StringVar()
        self.button_text.set("Start")
        self.algo_var = StringVar()
        self.all_cities = all_cities
        self.frame_select_cities = Frame(self.root)
        self.frame_select_cities.grid(row=1)
        self.frame_canvas = Frame(self.root)
        self.frame_canvas.grid(row=2)
        Label(self.root, text="Map of Romania", font="Times 13 bold").grid(row=0, columnspan=10)

    def create_checkboxes(self, side=LEFT, anchor=W):
        """To select cities which are to be a part of Traveling Salesman Problem"""

        row_number = 0
        column_number = 0

        for city in self.all_cities:
            var = IntVar()
            var.set(1)
            Checkbutton(self.frame_select_cities, text=city, variable=var).grid(
                row=row_number, column=column_number, sticky=W)

            self.vars.append(var)
            column_number += 1
            if column_number == 10:
                column_number = 0
                row_number += 1

    def create_buttons(self):
        """Create start and quit button"""

        Button(self.frame_select_cities, textvariable=self.button_text,
               command=self.run_traveling_salesman).grid(row=5, column=4, sticky=E + W)
        Button(self.frame_select_cities, text='Quit', command=self.on_closing).grid(
            row=5, column=5, sticky=E + W)

    def create_dropdown_menu(self):
        """Create dropdown menu for algorithm selection"""

        choices = {'Simulated Annealing', 'Genetic Algorithm', 'Hill Climbing'}
        self.algo_var.set('Simulated Annealing')
        dropdown_menu = OptionMenu(self.frame_select_cities, self.algo_var, *choices)
        dropdown_menu.grid(row=4, column=4, columnspan=2, sticky=E + W)
        dropdown_menu.config(width=19)

    def run_traveling_salesman(self):
        """Choose selected cities"""

        cities = []
        for i in range(len(self.vars)):
            if self.vars[i].get() == 1:
                cities.append(self.all_cities[i])

        tsp_problem = TSProblem(cities)
        self.button_text.set("Reset")
        self.create_canvas(tsp_problem)

    def calculate_canvas_size(self):
        """Width and height for canvas"""

        minx, maxx = sys.maxsize, -1 * sys.maxsize
        miny, maxy = sys.maxsize, -1 * sys.maxsize

        for value in romania_map.locations.values():
            minx = min(minx, value[0])
            maxx = max(maxx, value[0])
            miny = min(miny, value[1])
            maxy = max(maxy, value[1])

        # New locations squeezed to fit inside the map of romania
        for name, coordinates in romania_map.locations.items():
            self.frame_locations[name] = (coordinates[0] / 1.2 - minx +
                                          150, coordinates[1] / 1.2 - miny + 165)

        canvas_width = maxx - minx + 200
        canvas_height = maxy - miny + 200

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def create_canvas(self, problem):
        """creating map with cities"""

        map_canvas = Canvas(self.frame_canvas, width=self.canvas_width, height=self.canvas_height)
        map_canvas.grid(row=3, columnspan=10)
        current = Node(problem.initial)
        map_canvas.delete("all")
        self.romania_image = PhotoImage(file="../images/romania_map.png")
        map_canvas.create_image(self.canvas_width / 2, self.canvas_height / 2,
                                image=self.romania_image)
        cities = current.state
        for city in cities:
            x = self.frame_locations[city][0]
            y = self.frame_locations[city][1]
            map_canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   fill="red", outline="red")
            map_canvas.create_text(x - 15, y - 10, text=city)

        self.cost = StringVar()
        Label(self.frame_canvas, textvariable=self.cost, relief="sunken").grid(
            row=2, columnspan=10)

        self.speed = IntVar()
        speed_scale = Scale(self.frame_canvas, from_=500, to=1, orient=HORIZONTAL,
                            variable=self.speed, label="Speed ----> ", showvalue=0, font="Times 11",
                            relief="sunken", cursor="gumby")
        speed_scale.grid(row=1, columnspan=5, sticky=N + S + E + W)

        if self.algo_var.get() == 'Simulated Annealing':
            self.temperature = IntVar()
            temperature_scale = Scale(self.frame_canvas, from_=100, to=0, orient=HORIZONTAL,
                                      length=200, variable=self.temperature, label="Temperature ---->",
                                      font="Times 11", relief="sunken", showvalue=0, cursor="gumby")
            temperature_scale.grid(row=1, column=5, columnspan=5, sticky=N + S + E + W)
            self.simulated_annealing_with_tunable_T(problem, map_canvas)
        elif self.algo_var.get() == 'Genetic Algorithm':
            self.mutation_rate = DoubleVar()
            self.mutation_rate.set(0.05)
            mutation_rate_scale = Scale(self.frame_canvas, from_=0, to=1, orient=HORIZONTAL,
                                        length=200, variable=self.mutation_rate, label='Mutation Rate ---->',
                                        font='Times 11', relief='sunken', showvalue=0, cursor='gumby', resolution=0.001)
            mutation_rate_scale.grid(row=1, column=5, columnspan=5, sticky='nsew')
            self.genetic_algorithm(problem, map_canvas)
        elif self.algo_var.get() == 'Hill Climbing':
            self.no_of_neighbors = IntVar()
            self.no_of_neighbors.set(100)
            no_of_neighbors_scale = Scale(self.frame_canvas, from_=10, to=1000, orient=HORIZONTAL,
                                          length=200, variable=self.no_of_neighbors, label='Number of neighbors ---->',
                                          font='Times 11', relief='sunken', showvalue=0, cursor='gumby')
            no_of_neighbors_scale.grid(row=1, column=5, columnspan=5, sticky='nsew')
            self.hill_climbing(problem, map_canvas)

    def exp_schedule(k=100, lam=0.03, limit=1000):
        """One possible schedule function for simulated annealing"""

        return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

    def simulated_annealing_with_tunable_T(self, problem, map_canvas, schedule=exp_schedule()):
        """Simulated annealing where temperature is taken as user input"""

        current = Node(problem.initial)

        while True:
            T = schedule(self.temperature.get())
            if T == 0:
                return current.state
            neighbors = current.expand(problem)
            if not neighbors:
                return current.state
            next = random.choice(neighbors)
            delta_e = problem.value(next.state) - problem.value(current.state)
            if delta_e > 0 or probability(np.exp(delta_e / T)):
                map_canvas.delete("poly")

                current = next
                self.cost.set("Cost = " + str('%0.3f' % (-1 * problem.value(current.state))))
                points = []
                for city in current.state:
                    points.append(self.frame_locations[city][0])
                    points.append(self.frame_locations[city][1])
                map_canvas.create_polygon(points, outline='red', width=3, fill='', tag="poly")
                map_canvas.update()
                map_canvas.after(self.speed.get())

    def genetic_algorithm(self, problem, map_canvas):
        """Genetic Algorithm modified for the given problem"""

        def init_population(pop_number, gene_pool, state_length):
            """initialize population"""

            population = []
            for i in range(pop_number):
                population.append(utils.shuffled(gene_pool))
            return population

        def recombine(state_a, state_b):
            """recombine two problem states"""

            start = random.randint(0, len(state_a) - 1)
            end = random.randint(start + 1, len(state_a))
            new_state = state_a[start:end]
            for city in state_b:
                if city not in new_state:
                    new_state.append(city)
            return new_state

        def mutate(state, mutation_rate):
            """mutate problem states"""

            if random.uniform(0, 1) < mutation_rate:
                sample = random.sample(range(len(state)), 2)
                state[sample[0]], state[sample[1]] = state[sample[1]], state[sample[0]]
            return state

        def fitness_fn(state):
            """calculate fitness of a particular state"""

            fitness = problem.value(state)
            return int((5600 + fitness) ** 2)

        current = Node(problem.initial)
        population = init_population(100, current.state, len(current.state))
        all_time_best = current.state
        while True:
            population = [mutate(recombine(*select(2, population, fitness_fn)), self.mutation_rate.get())
                          for _ in range(len(population))]
            current_best = np.argmax(population, key=fitness_fn)
            if fitness_fn(current_best) > fitness_fn(all_time_best):
                all_time_best = current_best
                self.cost.set("Cost = " + str('%0.3f' % (-1 * problem.value(all_time_best))))
            map_canvas.delete('poly')
            points = []
            for city in current_best:
                points.append(self.frame_locations[city][0])
                points.append(self.frame_locations[city][1])
            map_canvas.create_polygon(points, outline='red', width=1, fill='', tag='poly')
            best_points = []
            for city in all_time_best:
                best_points.append(self.frame_locations[city][0])
                best_points.append(self.frame_locations[city][1])
            map_canvas.create_polygon(best_points, outline='red', width=3, fill='', tag='poly')
            map_canvas.update()
            map_canvas.after(self.speed.get())

    def hill_climbing(self, problem, map_canvas):
        """hill climbing where number of neighbors is taken as user input"""

        def find_neighbors(state, number_of_neighbors=100):
            """finds neighbors using two_opt method"""

            neighbors = []
            for i in range(number_of_neighbors):
                new_state = problem.two_opt(state)
                neighbors.append(Node(new_state))
                state = new_state
            return neighbors

        current = Node(problem.initial)
        while True:
            neighbors = find_neighbors(current.state, self.no_of_neighbors.get())
            neighbor = np.argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
            map_canvas.delete('poly')
            points = []
            for city in current.state:
                points.append(self.frame_locations[city][0])
                points.append(self.frame_locations[city][1])
            map_canvas.create_polygon(points, outline='red', width=3, fill='', tag='poly')
            neighbor_points = []
            for city in neighbor.state:
                neighbor_points.append(self.frame_locations[city][0])
                neighbor_points.append(self.frame_locations[city][1])
            map_canvas.create_polygon(neighbor_points, outline='red', width=1, fill='', tag='poly')
            map_canvas.update()
            map_canvas.after(self.speed.get())
            if problem.value(neighbor.state) > problem.value(current.state):
                current.state = neighbor.state
                self.cost.set("Cost = " + str('%0.3f' % (-1 * problem.value(current.state))))

    def on_closing(self):
        if messagebox.askokcancel('Quit', 'Do you want to quit?'):
            self.root.destroy()


if __name__ == '__main__':
    all_cities = []
    for city in romania_map.locations.keys():
        distances[city] = {}
        all_cities.append(city)
    all_cities.sort()

    # distances['city1']['city2'] contains euclidean distance between their coordinates
    for name_1, coordinates_1 in romania_map.locations.items():
        for name_2, coordinates_2 in romania_map.locations.items():
            distances[name_1][name_2] = np.linalg.norm(
                [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])
            distances[name_2][name_1] = np.linalg.norm(
                [coordinates_1[0] - coordinates_2[0], coordinates_1[1] - coordinates_2[1]])

    root = Tk()
    root.title("Traveling Salesman Problem")
    cities_selection_panel = TSPGui(root, all_cities)
    cities_selection_panel.create_checkboxes()
    cities_selection_panel.create_buttons()
    cities_selection_panel.create_dropdown_menu()
    root.protocol('WM_DELETE_WINDOW', cities_selection_panel.on_closing)
    root.mainloop()
