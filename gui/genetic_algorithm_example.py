# author: ad71
# A simple program that implements the solution to the phrase generation problem using
# genetic algorithms as given in the search.ipynb notebook.
# 
# Type on the home screen to change the target phrase
# Click on the slider to change genetic algorithm parameters
# Click 'GO' to run the algorithm with the specified variables
# Displays best individual of the current generation
# Displays a progress bar that indicates the amount of completion of the algorithm
# Displays the first few individuals of the current generation

import sys
import time
import random
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tkinter import *
from tkinter import ttk

import search
from utils import argmax

LARGE_FONT = ('Verdana', 12)
EXTRA_LARGE_FONT = ('Consolas', 36, 'bold')

canvas_width = 800
canvas_height = 600

black = '#000000'
white = '#ffffff'
p_blue = '#042533'
lp_blue = '#0c394c'

# genetic algorithm variables
# feel free to play around with these
target = 'Genetic Algorithm' # the phrase to be generated
max_population = 100 # number of samples in each population
mutation_rate = 0.1 # probability of mutation
f_thres = len(target) # fitness threshold
ngen = 1200 # max number of generations to run the genetic algorithm

generation = 0 # counter to keep track of generation number

u_case = [chr(x) for x in range(65, 91)] 		# list containing all uppercase characters
l_case = [chr(x) for x in range(97, 123)]		# list containing all lowercase characters
punctuations1 = [chr(x) for x in range(33, 48)]	# lists containing punctuation symbols
punctuations2 = [chr(x) for x in range(58, 65)]
punctuations3 = [chr(x) for x in range(91, 97)]
numerals = [chr(x) for x in range(48, 58)]		# list containing numbers

# extend the gene pool with the required lists and append the space character
gene_pool = []
gene_pool.extend(u_case)
gene_pool.extend(l_case)
gene_pool.append(' ')

# callbacks to update global variables from the slider values
def update_max_population(slider_value):
	global max_population
	max_population = slider_value

def update_mutation_rate(slider_value):
	global mutation_rate
	mutation_rate = slider_value

def update_f_thres(slider_value):
	global f_thres
	f_thres = slider_value

def update_ngen(slider_value):
	global ngen
	ngen = slider_value

# fitness function
def fitness_fn(_list):
	fitness = 0
	# create string from list of characters
	phrase = ''.join(_list)
	# add 1 to fitness value for every matching character
	for i in range(len(phrase)):
		if target[i] == phrase[i]:
			fitness += 1
	return fitness

# function to bring a new frame on top
def raise_frame(frame, init=False, update_target=False, target_entry=None, f_thres_slider=None):
	frame.tkraise()
	global target
	if update_target and target_entry is not None:
		target = target_entry.get()
		f_thres_slider.config(to=len(target))
	if init:
		population = search.init_population(max_population, gene_pool, len(target))
		genetic_algorithm_stepwise(population)

# defining root and child frames
root = Tk()
f1 = Frame(root)
f2 = Frame(root)

# pack frames on top of one another
for frame in (f1, f2):
	frame.grid(row=0, column=0, sticky='news')

# Home Screen (f1) widgets
target_entry = Entry(f1, font=('Consolas 46 bold'), exportselection=0, foreground=p_blue, justify=CENTER)
target_entry.insert(0, target)
target_entry.pack(expand=YES, side=TOP, fill=X, padx=50)
target_entry.focus_force()

max_population_slider = Scale(f1, from_=3, to=1000, orient=HORIZONTAL, label='Max population', command=lambda value: update_max_population(int(value)))
max_population_slider.set(max_population)
max_population_slider.pack(expand=YES, side=TOP, fill=X, padx=40)

mutation_rate_slider = Scale(f1, from_=0, to=1, orient=HORIZONTAL, label='Mutation rate', resolution=0.0001, command=lambda value: update_mutation_rate(float(value)))
mutation_rate_slider.set(mutation_rate)
mutation_rate_slider.pack(expand=YES, side=TOP, fill=X, padx=40)

f_thres_slider = Scale(f1, from_=0, to=len(target), orient=HORIZONTAL, label='Fitness threshold', command=lambda value: update_f_thres(int(value)))
f_thres_slider.set(f_thres)
f_thres_slider.pack(expand=YES, side=TOP, fill=X, padx=40)

ngen_slider = Scale(f1, from_=1, to=5000, orient=HORIZONTAL, label='Max number of generations', command=lambda value: update_ngen(int(value)))
ngen_slider.set(ngen)
ngen_slider.pack(expand=YES, side=TOP, fill=X, padx=40)

button = ttk.Button(f1, text='RUN', command=lambda: raise_frame(f2, init=True, update_target=True, target_entry=target_entry, f_thres_slider=f_thres_slider)).pack(side=BOTTOM, pady=50)

# f2 widgets
canvas = Canvas(f2, width=canvas_width, height=canvas_height)
canvas.pack(expand=YES, fill=BOTH, padx=20, pady=15)
button = ttk.Button(f2, text='EXIT', command=lambda: raise_frame(f1)).pack(side=BOTTOM, pady=15)

# function to run the genetic algorithm and update text on the canvas
def genetic_algorithm_stepwise(population):
	root.title('Genetic Algorithm')
	for generation in range(ngen):
		# generating new population after selecting, recombining and mutating the existing population
		population = [search.mutate(search.recombine(*search.select(2, population, fitness_fn)), gene_pool, mutation_rate) for i in range(len(population))]
		# genome with the highest fitness in the current generation
		current_best = ''.join(argmax(population, key=fitness_fn))
		# collecting first few examples from the current population
		members = [''.join(x) for x in population][:48]

		# clear the canvas
		canvas.delete('all')
		# displays current best on top of the screen
		canvas.create_text(canvas_width / 2, 40, fill=p_blue, font='Consolas 46 bold', text=current_best)

		# displaying a part of the population on the screen
		for i in range(len(members) // 3):
			canvas.create_text((canvas_width * .175), (canvas_height * .25 + (25 * i)), fill=lp_blue, font='Consolas 16', text=members[3 * i])
			canvas.create_text((canvas_width * .500), (canvas_height * .25 + (25 * i)), fill=lp_blue, font='Consolas 16', text=members[3 * i + 1])
			canvas.create_text((canvas_width * .825), (canvas_height * .25 + (25 * i)), fill=lp_blue, font='Consolas 16', text=members[3 * i + 2])

		# displays current generation number
		canvas.create_text((canvas_width * .5), (canvas_height * 0.95), fill=p_blue, font='Consolas 18 bold', text=f'Generation {generation}')

		# displays blue bar that indicates current maximum fitness compared to maximum possible fitness
		scaling_factor = fitness_fn(current_best) / len(target)
		canvas.create_rectangle(canvas_width * 0.1, 90, canvas_width * 0.9, 100, outline=p_blue)
		canvas.create_rectangle(canvas_width * 0.1, 90, canvas_width * 0.1 + scaling_factor * canvas_width * 0.8, 100, fill=lp_blue)
		canvas.update()

		# checks for completion
		fittest_individual = search.fitness_threshold(fitness_fn, f_thres, population)
		if fittest_individual:
			break

raise_frame(f1)
root.mainloop()