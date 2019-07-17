
# TODO: Logging and playback

from types import MethodType
import tkinter as tk
import PIL.Image as Image
import PIL.ImageTk as itk

from objects import Object
from utils import *
from agents import Agent

# Relative (%) offset values for image positioning from upper left corner of the cell
IMAGE_X_OFFSET = 0.5
IMAGE_Y_OFFSET = 0.5

# Relative (%) offset values for image positioning from the above image offset if there is an ID value set
IMAGEID_X_OFFSET = 0.05
IMAGEID_Y_OFFSET = 0.05

# Relative (%) offset values for the ID text field from the upper left corner of the cell
ID_X_OFFSET = 0.03
ID_Y_OFFSET = 0.0

#______________________________________________________________________________

class Icon():
    def __repr__(self): # Define the string representation of the class (this is to avoid importing the Object class
        return '<%s>' % getattr(self, '__name__', self.__class__.__name__)

    def __init__(self, parent, ef, images = {}):
        self.parent = parent        # parent object
        self.ef = ef
        self.images = images
        # TODO: Implement offsets in the icon class

    def object_to_image(self):
        if hasattr(self.parent, 'heading'):
            return self.ef.file2image[self.ef.class2file.get(getattr(self.parent, '__name__', self.parent.__class__.__name__),'') % self.ef.orientation[self.parent.heading]]
        else:
            return self.ef.file2image[self.ef.class2file.get(getattr(self.parent, '__name__', self.parent.__class__.__name__),'')]

    def move_to(self, newLocation):
        old_loc = self.ef.canvas.coords(self.images['image'])
        dx = (self.ef.cellwidth + 1) * (newLocation[0] - int(old_loc[0]/(self.ef.cellwidth+1)))
        dy = (self.ef.cellwidth + 1) * (newLocation[1] - int(old_loc[1]/(self.ef.cellwidth+1)))
        for img in self.images.values():
            self.ef.canvas.move(img, dx, dy)

    def rotate(self):
        if isinstance(self.parent, Agent): self.ef.canvas.itemconfig(self.parent.icon.images['image'], image=self.ef.object_to_image(self.parent))

    def hide(self):
        for img in self.images.values():
            self.ef.canvas.itemconfig(img, state='hidden')

    def destroy_images(self):
        for imgname in self.images.keys():
            self.ef.canvas.delete(self.images[imgname])
        self.images = {}

    def update(self):
        if isinstance(self.parent.location, tuple):
            self.rotate()
            self.move_to(self.parent.location)
        else:
            self.hide()

class EnvFrame(tk.Frame):
    def __init__(self, env, root = tk.Tk(), title='Robot Vacuum Simulation', cellwidth=50, n=10):
        update(self, cellwidth=cellwidth, running=False, delay=1.0)
        self.root = root
        self.running = 0
        self.delay = 0.1
        self.env = env
        self.cellwidth = cellwidth

        tk.Frame.__init__(self, None, width=min((cellwidth + 2) * env.width,self.root.winfo_screenwidth()),
                          height=min((cellwidth + 2) * env.height, self.root.winfo_screenheight()))
        self.root.title(title) 

        # Toolbar
        toolbar = tk.Frame(self, relief='raised', bd=2)
        toolbar.pack(side='top', fill='x')
        for txt, cmd in [('Step >', self.next_step), ('Run >>', self.run),
                         ('Stop [ ]', self.stop)]:
            tk.Button(toolbar, text=txt, command=cmd).pack(side='left')
        tk.Label(toolbar, text='Delay').pack(side='left')
        scale = tk.Scale(toolbar, orient='h', from_=0.0, to=3.0, resolution=0.1, length=300,
                         command=lambda d: setattr(self, 'delay', d))
        scale.set(self.delay)
        scale.pack(side='left')


        # Canvas for drawing on
        self.canvas = tk.Canvas(self, width=(cellwidth + 1) * env.width,
                                height=(cellwidth + 1) * env.height, background='white')

        hbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        hbar.config(command=self.canvas.xview)
        vbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(width=(cellwidth + 1) * env.width, height=(cellwidth + 1) * env.height)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)


        # Canvas click handlers (1 = left, 2 = middle, 3 = right)
        self.canvas.bind('<Button-1>', self.left_click)  ## What should this do?
        self.canvas.bind('<Button-2>', self.middle_click)
        self.canvas.bind('<Button-3>', self.right_click)
        if cellwidth:
            c = self.canvas
            for i in range(1, env.width + 1):
                c.create_line(0, i * (cellwidth + 1), env.height * (cellwidth + 1), i * (cellwidth + 1))
                c.pack(expand=1, fill='both')
            for j in range(1,env.height + 1):
                c.create_line(j * (cellwidth + 1), 0, j * (cellwidth + 1), env.width * (cellwidth + 1))
                c.pack(expand=1, fill='both')
        self.pack()

        self.class2file = {'':'', 'RandomReflexAgent':'robot-%s',
                       'Dirt':'dirt',
                       'Wall':'wall',
                        'Fire':'fire'}
        self.file2image = {'':None, 'robot-right':itk.PhotoImage(Image.open('img/robot-right.png').resize((int(0.8*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS)),
                       'robot-left':itk.PhotoImage(Image.open('img/robot-left.png').resize((int(0.8*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS)),
                       'robot-up':itk.PhotoImage(Image.open('img/robot-up.png').resize((int(0.8*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS)),
                       'robot-down':itk.PhotoImage(Image.open('img/robot-down.png').resize((int(0.8*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS)),
                       'dirt':itk.PhotoImage(Image.open('img/dirt.png').resize((int(0.8*cellwidth),int(0.4*cellwidth)),resample=Image.LANCZOS)),
                       'wall':itk.PhotoImage(Image.open('img/wall.png').resize((int(0.8*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS)),
                       'fire':itk.PhotoImage(Image.open('img/fire.png').resize((int(0.55*cellwidth),int(0.8*cellwidth)),resample=Image.LANCZOS))}
        # note up and down are switched, since (0,0) is in the upper left
        self.orientation = {(1,0): 'right', (-1,0): 'left', (0,-1): 'up', (0,1): 'down'}

        self.canvas.config(scrollregion=(0, 0, (self.cellwidth + 1) * self.env.width, (self.cellwidth + 1) * self.env.height))

    def background_run(self):
        #with Timer(name='Loop Timer', format='%.4f'):
            if self.running:
                self.env.step()
                self.update_display()

                ms = int(1000 * max(float(self.delay), 0.01))
                self.after(ms, self.background_run)

    def run(self):
        print('run')
        self.running = 1
        self.background_run()

    def next_step(self):
        self.env.step()
        self.update_display()

    def stop(self):
        print('stop')
        self.running = 0

    def left_click(self, event):
        loc = (int(event.x / (self.cellwidth + 1)), int(event.y / (self.cellwidth + 1)))
        objs = self.env.find_at(Object, loc)
        if not objs:
            obj_string = 'Nothing'
        else:
            obj_string = str([str(o)[:len(str(o))-1] + ' performance=%s>' % o.performance if hasattr(o, 'performance') else str(o) for o in objs])
        print('Cell (%s, %s) contains %s' %  (loc[0], loc[1], obj_string))

    def middle_click(self, event):
        pass

    def right_click(self, event):  # TODO: Add additional debugging for the Agent state
        loc = (int(event.x / (self.cellwidth + 1)), int(event.y / (self.cellwidth + 1)))
        agts = self.env.find_at(Agent, loc)
        if agts:
            for a in agts:
                hld = ''
                if hasattr(a, 'holding') and a.holding:
                    hld = a.holding
                else:
                    hld = 'Nothing'
                print('%s in Cell (%s, %s) is holding %s' % (a, loc[0], loc[1], hld))
        else:
            print('Cell (%s, %s) contains %s' % (loc[0], loc[1], 'No Agents'))

    def object_to_image(self,obj):
        if hasattr(obj, 'heading'):
            return self.file2image[self.class2file.get(getattr(obj, '__name__', obj.__class__.__name__),'') % self.orientation[obj.heading]]
        else:
            return self.file2image[self.class2file.get(getattr(obj, '__name__', obj.__class__.__name__),'')]

    def display_object(self, obj):
        obj.icon = self.NewIcon(obj)

        old_destroy = obj.destroy  # save old obj.destroy() method

        def destroy_with_images(self):  # define a new obj.destroy() method
            old_destroy()  # first run old obj.destroy()
            obj.icon.destroy_images()

        obj.destroy = MethodType(destroy_with_images, obj)
        return obj

    def NewIcon(self, obj):
        # lookup default image and add it to the list
        imgs = {}
        imgs['image'] = (self.canvas.create_image(
            (obj.location[0] + (IMAGE_X_OFFSET + (obj.id != '') * IMAGEID_X_OFFSET)) * (self.cellwidth + 1),
            (obj.location[1] + (IMAGE_Y_OFFSET + (obj.id != '') * IMAGEID_Y_OFFSET)) * (self.cellwidth + 1),
            image=self.object_to_image(obj), tag=getattr(obj, '__name__', obj.__class__.__name__)))

        if obj.id:
            imgs['id'] = (self.canvas.create_text((obj.location[0] + ID_X_OFFSET) * (self.cellwidth + 1),
                                               (obj.location[1] + ID_Y_OFFSET) * (self.cellwidth + 1),
                                               text=obj.id, anchor='nw', font=('Helvetica', int(self.cellwidth / 5.0)),
                                               tag=getattr(obj, '__name__', obj.__class__.__name__)))
        return Icon(obj, self, imgs)

    def configure_display(self):
        for obj in self.env.objects:
            obj = self.display_object(obj)
        # for i in range(len(self.env.objects)):
        #     self.env.objects[i] = self.display_object(self.env.objects[i])
        self.update_display()

    def update_display(self):
        for obj in self.env.objects:
            if hasattr(obj, 'icon') and obj.icon:
                obj.icon.update()
            else:
                self.display_object(obj)

        self.canvas.tag_lower('Dirt')
#______________________________________________________________________________
