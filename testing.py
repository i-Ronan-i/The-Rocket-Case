import random


def mutate(pop, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max): 
    """Takes current population member and add a probability chance to the PID parameters
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%. 
    However, for the weights only one is mutated if picked and the others moulded around that."""
    pop_curr = pop.copy()
    for i in range(0, len(pop_curr)):
        for o in range(len(pop_curr[i])-3) :
            if random.random() <= mut_prob:
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.95, 2) #Maintains 2 d.p
                else :
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.05, 2)
                    if pop_curr[i][0] > kp_max or pop_curr[i][1] > kp_max or pop_curr[i][2] > kp_max:
                        pop_curr[i][o] = float(kp_max) 
                    if pop_curr[i][3] > ki_max or pop_curr[i][4] > ki_max or pop_curr[i][5] > ki_max :
                        pop_curr[i][o] = float(ki_max)
                    if pop_curr[i][6] > kd_max or pop_curr[i][7] > kd_max or pop_curr[i][8] > kd_max :
                        pop_curr[i][o] = float(kd_max)
        #Weight Mutation.               
        weight_mut = False
        for o in range(len(pop_curr[i])-3, len(pop_curr[i])):
            if weight_mut == False:
                if random.random() <= mut_prob:
                    weight_mut = True
                    if random.random() < 0.5:
                        pop_curr[i][o] = round(pop_curr[i][o] * 0.95, 2) #Maintains 2 d.p
                    else:
                        pop_curr[i][o] = round(pop_curr[i][o] * 1.05, 2)
                if weight_mut == True:
                    weighed = 0
                    weigh_again = True
                    while weigh_again == True:
                        weigh_again = False
                        w_r = pop_curr[i][9]
                        w_v = pop_curr[i][10]
                        w_m = pop_curr[i][11]
                        l = round((w_r + w_v + w_m), 2)
                        w_r = round(w_r/l, 2)
                        w_v = round(w_v/l, 2)
                        w_m = round(w_m/l, 2)
                        pop_curr[i][9] = round(w_r, 2)
                        pop_curr[i][10] = round(w_v, 2)
                        pop_curr[i][11] = round(w_m, 2)

                        if round(pop_curr[i][9]+pop_curr[i][10]+pop_curr[i][11], 3) !=1.0 :
                            weigh_again = True
                            weighed += 1
                            if weighed > 2 and l != 1:
                                pop_curr[i][11] += 0.01
                            if weighed > 15:
                                print("Weighed loop stuck")
                    
                if round(pop_curr[i][9]+pop_curr[i][10]+pop_curr[i][11], 3) != 1.0:
                    print("M U T A T I O N      F A I L")
    return pop_curr

def create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Creates the initial population of the genetic algorithm while making sure it adheres to force constraints"""
    for s in range(pop_num):
        #Creating the random PID values
        kp_r = round(random.uniform(kp_min, kp_max), 2)
        kp_v = round(random.uniform(kp_min, kp_max), 2)
        kp_m = round(random.uniform(kp_min, kp_max), 2)
        ki_r = round(random.uniform(ki_min, ki_max), 2)
        ki_v = round(random.uniform(ki_min, ki_max), 2)
        ki_m = round(random.uniform(ki_min, ki_max), 2)
        kd_r = round(random.uniform(kd_min, kd_max), 2)
        kd_v = round(random.uniform(kd_min, kd_max), 2)
        kd_m = round(random.uniform(kd_min, kd_max), 2)

        #creating the weights 
        w_r = round(random.random(), 2)
        w_v = round(random.uniform(0, 1-w_r), 2)
        w_m = round(1 - w_r - w_v, 2)

        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kp_r, kp_v, kp_m, ki_r, ki_v, ki_m, kd_r, kd_v, kd_m, w_r, w_v, w_m])
    return pop


def crossover(a, b):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    new_a = []  #Clearing previous 
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut

    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]

    #Creates the new crossed-over list
    new_a = new_a1 + new_a2

    # Weight Check #
    ################
    if new_a[9]+new_a[10]+new_a[11] != 1.0:
        weighed = 0
        weigh_again = True
        while weigh_again == True:
            weigh_again = False
            w_r = new_a[9]
            w_v = new_a[10]
            w_m = new_a[11]
            l = round((w_r + w_v + w_m), 2)
            w_r = round(w_r/l, 2)
            w_v = round(w_v/l, 2)
            w_m = round(w_m/l, 2)
            new_a[9] = w_r
            new_a[10] = w_v
            new_a[11] = w_m

            if round(w_r + w_v + w_m, 3) != 1.0 :
                weigh_again = True
                weighed += 1
                if weighed > 2 and l != 1.0:
                    new_a[11] += 0.01
                if weighed > 15:
                    print("Crossover Weigh loop stuck")
    
    if round(new_a[9]+new_a[10]+new_a[11], 3) != 1.0:
        print("C R O S S O V E R      F A I L")
    return new_a

pop = []
pop_num = 60
mut_prob = 0.3

pop = create_initial(pop_num, pop, 0, 500, 0, 500, 0, 500)

#Crossover performed in top 20
pop_cross = []
for n in range(59):
    new_pop1 = crossover(pop[n], pop[n+1])
    pop_cross.append(new_pop1)

#Adds all currently available members
#Then mutates them.
pop_new = []
pop_premut = []
pop_premut = pop_cross
pop_new = mutate(pop_premut, mut_prob, 0, 500, 0, 500, 0, 500)