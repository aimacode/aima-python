from mimetypes import init
from search import *
from timeit import default_timer as timer
from datetime import timedelta
import time
def switch(type_of_problem):
    if type_of_problem=="queen":
        erfolgreiche_laufe=0
        succeded_gen = 0
        aes = 0
        start = time.time()
        for i in range(1):
            x,a = problem(100,8,fitness_fn_queens,[1,2,3,4,5,6,7,8],500,28,0.1)
            print(x,fitness_fn_queens(x))
            if(fitness_fn_queens(x)==28):
                succeded_gen += a
                erfolgreiche_laufe += 1
        if erfolgreiche_laufe!=0:
            aes = succeded_gen/erfolgreiche_laufe
        sr = erfolgreiche_laufe/100
        end = time.time()
        ausfuhrungszeit = end - start
        ##init_pop = 5
        ##pmut = 0.125
        ##ngen = 1000
        
        print(
        "(AES)Average Evaluations to a Solution(Geschwindigkeit) ist {}".format(aes)+
        "\n(SR)Success Rate(Lösungswahrscheinlichkeit) ist {}".format(sr)+
        "\nAusführungszeit ist {}".format(ausfuhrungszeit))
    else:
        erfolgreiche_laufe=0
        succeded_gen = 0
        aes = 0
        start = time.time()
        for i in range(100):
            x,a = problem(3,6,fitness_fn_landkart,["red","blue","grey"],200,6,0.01)
            print(x,fitness_fn_landkart(x))
            if(fitness_fn_landkart(x)==6):
                succeded_gen += a
                erfolgreiche_laufe += 1
        if erfolgreiche_laufe!=0:
            aes = succeded_gen/erfolgreiche_laufe
        sr = erfolgreiche_laufe/100
        end = time.time()
        ausfuhrungszeit = end - start
        init_pop = 5
        pmut = 0.125
        ngen = 1000
        
        print("Einstellungen -> init_pop = {}, gen = {}, mut = {}\n".format(init_pop,ngen,pmut)+
        "(AES)Average Evaluations to a Solution(Geschwindigkeit) ist {}".format(aes)+
        "\n(SR)Success Rate(Lösungswahrscheinlichkeit) ist {}".format(sr)+
        "\nAusführungszeit ist {}".format(ausfuhrungszeit))    
def main():
    switch("queen")

    
def problem(init_p,state_length,fitness_fn,gene_pool,ngen,f_thres,pmut):
    population = init_population(init_p,gene_pool,state_length)
    x,a = genetic_algorithm(population = population,fitness_fn=fitness_fn,gene_pool=gene_pool,ngen=ngen,f_thres= f_thres,pmut= pmut)
    return x,a

def fitness_fn_queens(individuum):
    fitness_ct = 28
    for idx, x in enumerate(individuum):
        for idx2,x2 in enumerate(individuum):
            if(idx>=idx2):
                continue
            else:
                if(abs(idx-idx2)==abs(x-x2)): # 2,4    5,1   
                    fitness_ct-=1
                if(x==x2):
                    fitness_ct-=1
    
    return fitness_ct

def fitness_fn_landkart(individuum):
    fitness_ct = 6
    if(individuum[1]==individuum[2]):
        fitness_ct -= 1
    if(individuum[0]==individuum[1] or individuum[0]==individuum[2]):
        if(individuum[0]==individuum[1]==individuum[2]):
            fitness_ct -= 2
        else:
            fitness_ct -= 1
    if(individuum[1]==individuum[3] or individuum[2]==individuum[3]):
        if(individuum[1]==individuum[2]==individuum[3]):
            fitness_ct -= 2
        else:
            fitness_ct -= 1
    if(individuum[3]==individuum[4]):
        fitness_ct-=1
    return fitness_ct
   
if __name__ == "__main__":
    main()


