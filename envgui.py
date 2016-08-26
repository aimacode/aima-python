# ______________________________________________________________________________
# GUI - Graphical User Interface for Environments
# If you do not have Tkinter installed, either get a new installation of Python
# (Tkinter is standard in all new releases), or delete the rest of this file
# and muddle through without a GUI.
#
# Excerpted from:
# http://web.media.mit.edu/~havasi/MAS.S60/code/lemmatizer_learning/aima/agents.py
# Catherine Havasi
# 2012-04-10
#
# Revised:
# William H. Hooper
# 2016-07-13
# Python 2 -> 3

import tkinter as tk  # pip install tkinter
from tkinter import ttk
from PIL import ImageTk, Image  # pip install pillow
import os

class EnvGUI(tk.Tk, object):
    def __init__(self, env, title='AIMA GUI', cellsize=200, n=10):
        # Initialize window

        super(EnvGUI, self).__init__()
        self.title(title)

        # Create components
        w = env.width
        h = env.height
        self.canvas = EnvCanvas(self, env, cellsize, w, h)
        toolbar = EnvToolbar(self, env, self.canvas)
        for w in [self.canvas, toolbar]:
            w.pack(side="bottom", fill="x", padx="3", pady="3")

    def getCanvas(self):
        return self.canvas

class EnvToolbar(tk.Frame, object):
    def __init__(self, parent, env, canvas):
        super(EnvToolbar, self).__init__(parent, relief='raised', bd=2)

        # Initialize instance variables

        self.env = env
        self.canvas = canvas
        self.running = False
        self.speed = 1.0

        # Create buttons and other controls

        for txt, cmd in [#('Step >', self.env.step),
                         ('Step >', self.step),
                         ('Run >>', self.run),
                         ('Stop [ ]', self.stop),
                         ('List things', self.list_things),
                         ('List agents', self.list_agents)]:
            ttk.Button(self, text=txt, command=cmd).pack(side='left')

        tk.Label(self, text='Speed').pack(side='left')
        scale = tk.Scale(self, orient='h',
                         from_=(1.0), to=100.0, resolution=1.0,
                         command=self.set_speed)
        scale.set(self.speed)
        scale.pack(side='left')

    def step(self):
        self.env.step()
        self.canvas.update()

    def run(self):
        print('run')
        self.running = True
        self.background_run()

    def stop(self):
        print('stop')
        self.running = False

    def background_run(self):
        if self.running:
            self.step()
            # ms = int(1000 * max(float(self.speed), 0.5))
            # ms = max(int(1000 * float(self.delay)), 1)
            delay_sec = 10.0 / max(self.speed, 1.0)  # avoid division by zero
            ms = int(1000.0 * delay_sec)  # seconds to milliseconds
            self.after(ms, self.background_run)

    def list_things(self):
        print("Things in the environment:")
        for obj in self.env.things:
            print("%s at %s" % (obj, obj.location))

    def list_agents(self):
        print("Agents in the environment:")
        for agt in self.env.agents:
            print("%s at %s" % (agt, agt.location))

    def set_speed(self, speed):
        self.speed = float(speed)

class Empty:
    pass

class EnvCanvas(tk.Canvas, object):
    def __init__(self, parent, env, cellwidth, w, h):
        self.env = env
        cellheight = cellwidth
        canvwidth = cellwidth * w  # (cellwidth + 1 ) * n
        canvheight = cellheight * h  # (cellwidth + 1) * n
        super(EnvCanvas, self).__init__(parent, background="white",)

        # Initialize instance variables
        self.env = env
        self.cellwidth = cellwidth
        self.cellheight = cellheight
        self.w = w
        self.h = h
        # print(
        #     "cellwidth, canvwidth, camvheight = %d, %d, %d" % \
        #     (self.cellwidth, canvwidth, canvheight))

        # Set up image dictionary.
        # Ugly hack: we need to keep a reference to each ImageTk.PhotoImage,
        # or it will be garbage collected.  This dictionary maps image files
        # that have been opened to their PhotoImage objects
        self.fnMap = { Empty: 'images/default.png'}
        self.images = {}
        cwd = os.getcwd()
        default = self.get_image(self.fnMap[Empty])

        self.cells = [[0 for x in range(w)] for y in range(h)]
        for x in range(w):
            for y in range(h):
                cell = ttk.Frame(self)
                contents = ttk.Label(cell, image=default)
                contents.pack(side="bottom", fill="both", expand="yes")
                cell.grid(row=y, column=x)
                self.cells[y][x] = cell

        # Bind canvas events.

        # self.bind('<Button-1>', self.user_left) ## What should this do?
        # self.bind('<Button-2>', self.user_edit_objects)
        # self.bind('<Button-3>', self.user_add_object)

        self.pack()

    def mapImageNames(self, fnMap):
        self.fnMap.update(fnMap)

    def get_image(self, concat):
        """concat = 'filename1[+filename2]'
        Try to find the image in the images dictionary.
        If it's not there: open each file, create it,
        and paste it over the composite.
        When all names are processed, stick the composite
        image in the dictionary, and return the image in
        a form usable by the canvas."""
        if concat in self.images:
            tk_image = self.images[concat]
        else:
            filenames = concat.split('+')
            fn0 = filenames[0]
            pil_image = Image.open(fn0)
            for fni in filenames[1:]:
                pi = Image.open(fni)
                #tki = ImageTk.PhotoImage(pi)
                pil_image.paste(pi, mask=pi)
            pil_image = pil_image.resize((self.cellwidth, self.cellheight),
                                         Image.ANTIALIAS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.images[concat] = tk_image
        return tk_image

    def update(self):
        '''Create a tiled image of the XY Environment,
        based on the things in each cell.'''
        env = self.env
        for x in range(self.w):
            for y in range(self.h):
                cell = self.cells[y][x]
                filenames = ''
                tList = env.list_things_at((x, y))
                for thing in tList:
                    tclass = thing.__class__
                    tname = self.fnMap[tclass]
                    if filenames == '':
                        filenames = tname
                    elif not tname in filenames:
                        filenames += '+' + tname
                if filenames == '':
                    filenames = self.fnMap[Empty]
                bg = self.get_image(filenames)
                # contents = ttk.Label(cell, image=bg)
                contents = cell.winfo_children()[0]
                contents.config(image=bg)
                contents.pack(side="bottom", fill="both", expand="yes")

    def user_left(self, event):
        print('left at %d, %d' % self.event_cell(event))

    def user_edit_objects(self, event):
        """Choose an object within radius and edit its fields."""
        pass

    def user_add_object(self, event):
        """Pops up a menu of Object classes; you choose the
        one you want to put in this square."""
        cell = self.event_cell(event)
        xy = self.cell_topleft_xy(cell)
        menu = tk.Menu(self, title='Edit (%d, %d)' % cell)
        # Generalize object classes available,
        # and why is self.run the command?
        # for (txt, cmd) in [('Wumpus', self.run), ('Pit', self.run)]:
        #    menu.add_command(label=txt, command=cmd)
        obj_classes = self.env.object_classes()

        def class_cmd(oclass):
            def cmd():
                obj = oclass()
                self.env.add_object(obj, cell)
                # what about drawing it on the canvas?
                print(
                    "Drawing object %s at %s %s" % (obj, cell, xy))
                tk_image = self.get_image(oclass.image_file)
                self.canvas.create_image(xy, anchor="nw", image=tk_image)

            return cmd

        for oclass in obj_classes:
            menu.add_command(label=oclass.__name__, command=class_cmd(oclass))

        menu.tk_popup(event.x + self.winfo_rootx(),
                      event.y + self.winfo_rooty())

    def event_cell(self, event):
        return self.xy_cell(event.x, event.y)

    def xy_cell(self, x, y):
        """Given an (x, y) on the canvas, return the row and column
        of the cell containing it."""
        w = self.cellwidth
        return x / w, y / w

    def cell_topleft_xy(self, row, column):
        """Given a (row, column) tuple, return the (x, y) coordinates
        of the cell(row, column)'s top left corner."""

        w = self.cellwidth
        return w * row, w * column