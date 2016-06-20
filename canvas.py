from IPython.display import HTML, display, clear_output

_canvas = """
<script type="text/javascript" src="./js/canvas.js"></script>
<div>
<canvas id="{0}" width="{1}" height="{2}" style="background:rgba(158, 167, 184, 0.2);" onclick='click_callback(this, event, "{3}")'></canvas>
</div>

<script> var {0}_canvas_object = new Canvas("{0}");</script>
"""

class Canvas:
    """Inherit from this class to manage the HTML canvas element in jupyter notebooks.
    To create an object of this class any_name_xyz = Canvas("any_name_xyz")
    The first argument given must be the name of the object being create
    IPython must be able to refernce the variable name that is being passed
    """

    def __init__(self, varname, id=None, width=800, height=600):
        """"""
        self.name = varname
        self.id = id or varname
        self.width = width
        self.height = height
        self.html = _canvas.format(self.id, self.width, self.height, self.name)
        self.exec_list = []
        display(HTML(self.html))

    def mouse_click(self, x, y):
        "Override this method to handle mouse click at position (x, y)"
        raise NotImplementedError

    def mouse_move(self, x, y):
        raise NotImplementedError

    def execute(self, exec_str):
        "Stores the command to be exectued to a list which is used later during update()"
        if not isinstance(exec_str, str):
            print("Invalid execution argument:", exec_str)
            self.alert("Recieved invalid execution command format")
        prefix = "{0}_canvas_object.".format(self.id)
        self.exec_list.append(prefix + exec_str + ';')

    def fill(self, r, g, b):
        "Changes the fill color to a color in rgb format"
        self.execute("fill({0}, {1}, {2})".format(r, g, b))

    def stroke(self, r, g, b):
        "Changes the colors of line/strokes to rgb"
        self.execute("stroke({0}, {1}, {2})".format(r, g, b))

    def strokeWidth(self, w):
        "Changes the width of lines/strokes to 'w' pixels"
        self.execute("strokeWidth({0})".format(w))

    def rect(self, x, y, w, h):
        "Draw a rectangle with 'w' width, 'h' height and (x, y) as the top-left corner"
        self.execute("rect({0}, {1}, {2}, {3})".format(x, y, w, h))

    def rect_n(self, xn, yn, wn, hn):
        "Similar to rect(), but the dimensions are normalized to fall between 0 and 1"
        x = round(xn * self.width)
        y = round(yn * self.height)
        w = round(wn * self.width)
        h = round(hn * self.height)
        self.rect(x, y, w, h)

    def line(self, x1, y1, x2, y2):
        "Draw a line from (x1, y1) to (x2, y2)"
        self.execute("line({0}, {1}, {2}, {3})".format(x1, y1, x2, y2))

    def line_n(self, x1n, y1n, x2n, y2n):
        "Similar to line(), but the dimensions are normalized to fall between 0 and 1"
        x1 = round(x1n * self.width)
        y1 = round(y1n * self.height)
        x2 = round(x2n * self.width)
        y2 = round(y2n * self.height)
        self.line(x1, y1, x2, y2)

    def arc(self, x, y, r, start, stop):
        "Draw an arc with (x, y) as centre, 'r' as radius from angles 'start' to 'stop'"
        self.execute("arc({0}, {1}, {2}, {3}, {4})".format(x, y, r, start, stop))

    def arc_n(self, xn ,yn, rn, start, stop):
        """Similar to arc(), but the dimensions are normalized to fall between 0 and 1
        The normalizing factor for radius is selected between width and height by seeing which is smaller
        """
        x = round(xn * self.width)
        y = round(yn * self.height)
        r = round(rn * min(self.width, self.height))
        self.arc(x, y, r, start, stop)

    def clear(self):
        "Clear the HTML canvas"
        self.execute("clear()")

    def font(self, font):
        "Changes the font of text"
        self.execute('font("{0}")'.format(font))

    def text(self, txt, x, y, fill=True):
        "Display a text at (x, y)"
        if fill:
            self.execute('fill_text("{0}", {1}, {2})'.format(txt, x, y))
        else:
            self.execute('stroke_text("{0}", {1}, {2})'.format(txt, x, y))

    def text_n(self, txt, xn, yn, fill=True):
        "Similar to text(), but with normalized coordinates"
        x = round(xn * self.width)
        y = round(yn * self.height)
        self.text(txt, x, y, fill)

    def alert(self, message):
        "Immediately display an alert"
        display(HTML('<script>alert("{0}")</script>'.format(message)))

    def update(self):
        "Execute the JS code to execute the commands queued by execute()"
        exec_code = "<script>\n" + '\n'.join(self.exec_list) + "\n</script>"
        self.exec_list = []
        display(HTML(exec_code))
