from IPython.display import HTML, display, clear_output
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
