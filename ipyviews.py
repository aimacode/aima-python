from IPython.display import HTML, display, clear_output
from collections import defaultdict
from agents import PolygonObstacle
import time
import json
import copy
import __main__


# ______________________________________________________________________________
# Continuous environment


_CONTINUOUS_WORLD_HTML = '''
<div>
    <canvas class="main-robo-world" width="{0}" height="{1}" style="background:rgba(158, 167, 184, 0.2);" data-world_name="{2}"  onclick="getPosition(this,event)"/>
</div>

<script type="text/javascript">
var all_polygons = {3};
{4}
</script>
'''  # noqa

with open('js/continuousworld.js', 'r') as js_file:
    _JS_CONTINUOUS_WORLD = js_file.read()


class ContinuousWorldView:
    """ View for continuousworld Implementation in agents.py """

    def __init__(self, world, fill="#AAA"):
        self.time = time.time()
        self.world = world
        self.width = world.width
        self.height = world.height

    def object_name(self):
        globals_in_main = {x: getattr(__main__, x) for x in dir(__main__)}
        for x in globals_in_main:
            if isinstance(globals_in_main[x], type(self)):
                if globals_in_main[x].time == self.time:
                    return x

    def handle_add_obstacle(self, vertices):
        """ Vertices must be a nestedtuple. This method
        is called from kernel.execute on completion of
        a polygon. """
        self.world.add_obstacle(vertices)
        self.show()

    def handle_remove_obstacle(self):
        return NotImplementedError

    def get_polygon_obstacles_coordinates(self):
        obstacle_coordiantes = []
        for thing in self.world.things:
            if isinstance(thing, PolygonObstacle):
                obstacle_coordiantes.append(thing.coordinates)
        return obstacle_coordiantes

    def show(self):
        clear_output()
        total_html = _CONTINUOUS_WORLD_HTML.format(self.width, self.height, self.object_name(),
                                                   str(self.get_polygon_obstacles_coordinates()),
                                                   _JS_CONTINUOUS_WORLD)
        display(HTML(total_html))


# ______________________________________________________________________________
# Grid environment

_GRID_WORLD_HTML = '''
<div class="map-grid-world" >
    <canvas data-world_name="{0}"></canvas>
    <div style="min-height:20px;">
        <span></span>
    </div>
</div>
<script type="text/javascript">
var gridArray = {1} , size = {2} , elements = {3};
{4}
</script>
'''

with open('js/gridworld.js', 'r') as js_file:
    _JS_GRID_WORLD = js_file.read()


class GridWorldView:
    """ View for grid world. Uses XYEnviornment in agents.py as model.
        world: an instance of XYEnviornment.
        block_size: size of individual blocks in pixes.
        default_fill: color of blocks. A hex value or name should be passed.
    """

    def __init__(self, world, block_size=30, default_fill="white"):
        self.time = time.time()
        self.world = world
        self.labels = defaultdict(str)  # locations as keys
        self.representation = {"default": {"type": "color", "source": default_fill}}
        self.block_size = block_size

    def object_name(self):
        globals_in_main = {x: getattr(__main__, x) for x in dir(__main__)}
        for x in globals_in_main:
            if isinstance(globals_in_main[x], type(self)):
                if globals_in_main[x].time == self.time:
                    return x

    def set_label(self, coordinates, label):
        """ Add lables to a particular block of grid.
            coordinates: a tuple of (row, column).
            rows and columns are 0 indexed.
        """
        self.labels[coordinates] = label

    def set_representation(self, thing, repr_type, source):
        """ Set the representation of different things in the
            environment.
            thing: a thing object.
            repr_type : type of representation can be either "color" or "img"
            source: Hex value in case of color. Image path in case of image.
        """
        thing_class_name = thing.__class__.__name__
        if repr_type not in ("img", "color"):
            raise ValueError('Invalid repr_type passed. Possible types are img/color')
        self.representation[thing_class_name] = {"type": repr_type, "source": source}

    def handle_click(self, coordinates):
        """ This method needs to be overidden. Make sure to include a
            self.show() call at the end. """
        self.show()

    def map_to_render(self):
        default_representation = {"val": "default", "tooltip": ""}
        world_map = [[copy.deepcopy(default_representation) for _ in range(self.world.width)]
                     for _ in range(self.world.height)]

        for thing in self.world.things:
            row, column = thing.location
            thing_class_name = thing.__class__.__name__
            if thing_class_name not in self.representation:
                raise KeyError('Representation not found for {}'.format(thing_class_name))
            world_map[row][column]["val"] = thing.__class__.__name__

        for location, label in self.labels.items():
            row, column = location
            world_map[row][column]["tooltip"] = label

        return json.dumps(world_map)

    def show(self):
        clear_output()
        total_html = _GRID_WORLD_HTML.format(
            self.object_name(), self.map_to_render(),
            self.block_size, json.dumps(self.representation), _JS_GRID_WORLD)
        display(HTML(total_html))
