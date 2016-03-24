from IPython.display import HTML, display, clear_output

_canvas = """
<script type="text/javascript" src="./canvas.js" />
<div>
<canvas id="{0}" width="{1}" height="{2}" style="background:rgba(158, 167, 184, 0.2);" onclick="click_callback(this, event)"/>
</div>

<script> var {0}_canvas_object = new Canvas("{0}");</script>
"""

class Canvas:
    """Use this to manage the HTML canvas element in jupyter notebooks"""

    def __init__(self, varname, id=None, width=800, height=600):
        self.name = varname
        self.id = id or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.id, self.width, self.height, self.name)
        self.exec_list = []
        display(HTML(self.html))

    def mouse_click(self, x, y):
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def fill(self, r, g, b):
        self.exec_list.append("{0}_canvas_object.fill({1}, {2}, {3});".format(self.id, r, g, b))

    def stroke(self, r, g, b):
        self.exec_list.append("{0}_canvas_object.stroke({1}, {2}, {3});".format(self.id, r, g, b))

    def rect(self, x, y, w, h):
        self.exec_list.append("{0}_canvas_object.rect({1}, {2}, {3}, {4});".format(self.id, x, y, w, h))   

    def line(self, x1, y1, x2, y2):
        self.exec_list.append("{0}_canvas_object.line({1}, {2}, {3}, {4});".format(self.id, x1, y1, x2, y2))

    def arc(self, x, y, r, start, stop):
        self.exec_list.append("{0}_canvas_object.arc({1}, {2}, {3}, {4}, {5});".format(self.id, x, y, r, start, stop))

    def update(self):        
        exec_code = "<script>\n"+'\n'.join(self.exec_list)+"\n</script>"
        self.exec_list = []
        display(HTML(exec_code))
