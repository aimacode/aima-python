from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Grassfire import Grassfire

# Initialize grid rows, columns, and obstacle probability.
rows = 8
cols = 8
obstProb = 0.3

# Instantiate Grassfire class. Initialize a grid and colorGrid.
Grassfire = Grassfire()
grid = Grassfire.random_grid(rows=rows, cols=cols, obstacleProb=obstProb)
colorGrid = Grassfire.color_grid(grid)

# Initialize figure, imshow object, and axis.
fig = plt.figure()
gridPlot = plt.imshow(colorGrid, interpolation='nearest')
ax = gridPlot._axes
ax.grid(visible=True, ls='solid', color='k', lw=1.5)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Initialize text annotations to display obstacle probability, rows, cols.
obstText = ax.annotate('', (0.15, 0.01), xycoords='figure fraction')
colText = ax.annotate('', (0.15, 0.04), xycoords='figure fraction')
rowText = ax.annotate('', (0.15, 0.07), xycoords='figure fraction')

def set_axis_properties(rows, cols):
    '''Set axis/imshow plot properties based on number of rows, cols.'''
    ax.set_xlim((0, cols))
    ax.set_ylim((rows, 0))
    ax.set_xticks(np.arange(0, cols+1, 1))
    ax.set_yticks(np.arange(0, rows+1, 1))
    gridPlot.set_extent([0, cols, 0, rows])

def update_annotations(rows, cols, obstProb):
    '''Update annotations with obstacle probability, rows, cols.'''
    obstText.set_text('Obstacle density: {:.0f}%'.format(obstProb * 100))
    colText.set_text('Rows: {:d}'.format(rows))
    rowText.set_text('Columns: {:d}'.format(cols))

set_axis_properties(rows, cols)
update_annotations(rows, cols, obstProb)

# Disable default figure key bindings.
fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

def on_key_press(event):
    '''Handle key presses as follows:
        Enter: Exit script.
        Shift: Randomize the start and dest cells for the current grid.
        Ctrl: Randomly generate a new grid based on the current values
            of "rows", "cols", and "obstProb".
        Right/Left: Increment/decrement value of "rows".
        Up/Down: Increment/decrement value of "cols".
        0-9: Set value of "obstProb" (obstacle probability) to key / 10,
            eg, pressing 4 would set obstProb = 0.4.
    '''
    global grid, rows, cols, obstProb
    if event.key == 'enter':
        ani._stop()
        exit()
    elif event.key == 'shift':
        Grassfire.set_start_dest(grid)
        Grassfire.reset_grid(grid)
        ani.frame_seq = ani.new_frame_seq()
    elif event.key == 'control':
        grid = Grassfire.random_grid(rows=rows, cols=cols, obstacleProb=obstProb)
        set_axis_properties(rows, cols)
        ani._iter_gen = Grassfire.find_path(grid)
    elif event.key == 'right':
        rows += 1
        update_annotations(rows, cols, obstProb)
    elif event.key == 'left' and rows > 1:
        rows -= 1
        update_annotations(rows, cols, obstProb)
    elif event.key == 'up':
        cols += 1
        update_annotations(rows, cols, obstProb)
    elif event.key == 'down' and cols > 1:
        cols -= 1
        update_annotations(rows, cols, obstProb)
    elif event.key.isdigit():
        obstProb = int(event.key) / 10
        update_annotations(rows, cols, obstProb)
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Functions init_anim() and update_anim() are for use with FuncAnimation.
def init_anim():
    '''Plot grid in its initial state by resetting "grid".'''
    Grassfire.reset_grid(grid)
    colorGrid = Grassfire.color_grid(grid)
    gridPlot.set_data(colorGrid)

def update_anim(dummyFrameArgument):
    '''Update plot based on values in "grid" ("grid" is updated
        by the generator--this function simply passes "grid" to
        the color_grid() function to get an image array).
    '''
    colorGrid = Grassfire.color_grid(grid)
    gridPlot.set_data(colorGrid)

# Create animation object. Supply generator function to frames.
ani = animation.FuncAnimation(fig, update_anim,
    init_func=init_anim, frames=Grassfire.find_path(grid),
    repeat=True, interval=150)

# Turn on interactive plotting and show figure.
plt.ion()
plt.show(block=True)
