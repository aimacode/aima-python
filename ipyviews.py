try:
    from IPython.display import HTML, display, clear_output
except ImportError:
    print('IPython not available.')

from agents import PolygonObstacle
import time
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
'''

with open('js/continuousworld.js', 'r') as js_file:
    _JS_CONTINUOUS_WORLD = js_file.read()


class ContinuousWorldView:
    ''' View for continuousworld Implementation in agents.py '''

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
        total_html = _CONTINUOUS_WORLD_HTML.format(self.width, self.height, self.object_name(), str(self.get_polygon_obstacles_coordinates()), _JS_CONTINUOUS_WORLD)
        display(HTML(total_html))
