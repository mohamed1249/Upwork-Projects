import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
from sympy.abc import x
from sympy.utilities.lambdify import lambdify
plt.rcParams.update({'font.size': 22})

# Collect given parameters into a dictionary
p = {
    'x' : np.linspace(-0.9,0.9), # Spatial Domain of problem
    'piA' : x**2, # First given function
    'piB' : x**3+x**2, # Second given function
    'TOL' : 10**-8, # Tolerance for newton's method
    'maxit' : 20, # Max number of newton iterations
    'x0' : np.array([-0.8, -0.4, -0.2, 0.8]), # Newton initial conditions
    'parents' : 12, # Number of Parents
    'S' : 50, # Number of design strings
    'G' : 100, # Number of generations
    'dv' : 1, # Number of design variables
    'SB' : np.array([-0.9,0.9]) # Search bounds for design variables
}
############################### Problem 2.1 ###################################
piAPlot = lambdify(x,p['piA'])
piBPlot = lambdify(x,p['piB'])
plt.figure(figsize = (15,7))
plt.plot(p['x'],piAPlot(p['x']))
plt.plot(p['x'],piBPlot(p['x']))
plt.title("Given Functions")
plt.xlabel('x')
plt.ylabel("$\Pi(x)$")
plt.legend(("$\Pi_a$", "$\Pi_b$"))
plt.show()
############################### Problem 2.2 ###################################

def myNewton(func, x0, TOL, maxit):
    its = 0
    f_df = func.diff(x)
    df = lambdify(x,f_df)
    f_ddf = f_df.diff(x)
    ddf = lambdify(x,f_ddf)
    hist = np.zeros([1, maxit])
    while its<maxit:
        sol = x0 - df(x0)/ddf(x0)
        if np.abs(sol)<TOL:
            break
        hist[0][its] = sol
        its = its + 1
        x0 = sol
    hist = hist[0][0:its]
    its = its+1
    return sol, its, np.array(hist,dtype=object)
####################### Problem 2.3.A (optimize PiA) ##########################
for i in range(p['x0'].size): # Loop through all given x0 values

    # Call newton function ... 
    if not os.path.isfile('resultA'+str(i)+'.pkl'):
        solA, itsA, histA = myNewton(p['piA'], p['x0'][i], p['TOL'], p['maxit'])

    # Saving your variables to a file (aka pickling)
    if os.path.isfile('resultA'+str(i)+'.pkl'):
        with open('resultA'+str(i)+'.pkl', 'rb') as f:
            solA = pickle.load(f)
            itsA = pickle.load(f)
            histA = pickle.load(f)

    else:
        with open('resultA'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(solA, f)
            pickle.dump(itsA, f)
            pickle.dump(histA, f)
   
    # Plot convergence
    plt.figure(figsize = (15,7))
    plt.plot(p['x'],piAPlot(p['x']))
    plt.plot(p['x0'][i],piAPlot(p['x0'])[i],'r*',ms = 10)
    plt.plot(histA,piAPlot(histA))
    plt.plot(solA,piAPlot(solA),'b*',ms = 10)
    plt.title("Convergence of Newton's Method in %d Iterations" % itsA)
    plt.xlabel('$x$')
    plt.ylabel('$\Pi_a(x)$')
    plt.legend(("$\Pi_a$","$x_0$ =  %f" % p['x0'][i],"Newton's Convergence", 'sol = %f' % solA))
    plt.show()
####################### Problem 2.3.B (optimize PiB) ######################


for i in range(p['x0'].size): # Loop through all given x0 values
    
    # Call newton function ... 
    if not os.path.isfile('resultB'+str(i)+'.pkl'):
        solB, itsB, histB = myNewton(p['piB'], p['x0'][i], p['TOL'], p['maxit'])

    # Saving your variables to a file (aka pickling)
    if os.path.isfile('resultB'+str(i)+'.pkl'):
        with open('resultB'+str(i)+'.pkl', 'rb') as f:
            solB = pickle.load(f)
            itsB = pickle.load(f)
            histB = pickle.load(f)

    else:
        with open('resultB'+str(i)+'.pkl', 'wb') as f:
            pickle.dump(solB, f)
            pickle.dump(itsB, f)
            pickle.dump(histB, f)
    
    # Plot convergence
    plt.figure(figsize = (15,7))
    plt.plot(p['x'],piBPlot(p['x']))
    plt.plot(p['x0'][i],piBPlot(p['x0'])[i],'r*',ms = 10)
    plt.plot(histB,piBPlot(histB))
    plt.plot(solB,piBPlot(solB),'b*',ms = 10)
    plt.title("Convergence of Newton's Method in %d Iterations" % itsB)
    plt.xlabel('$x$')
    plt.ylabel('$\Pi_b(x)$')
    plt.legend(("$\Pi_b$","$x_0$ =  %f" %p['x0'][i],"Newton's Convergence",'sol = %f' % solB))
    plt.show()
########################### Problem 2.4 #######################################

def myGA(parents, G, S, dv, func, SB):
    Pi = np.zeros(S) # All costs in an individual generation
    ff = lambdify(x,func)
    # Generate initial random population
    Lambda = np.zeros([G, S])
    Lam = (SB[1] - SB[0])*np.random.rand(dv,S) + SB[0]
    Orig = Lam
    # Initially, calculate cost for all strings. After, only calculate new strings since top P already calculated
    start = 0 
    
    for i in range(G): # Loop through generations
        
        # Calculate fitness of unknown design string costs
        for j in range(start,S): # Evaluate fitness of strings
            Pi[j] = ff(Lam[:,j])
            
        
        # Sort cost and design strings based on performance
        ind = np.argsort(Pi)
        Pi = np.sort(Pi)
        Lam = Lam[:,ind]
      
        # Generate offspring radnom parameters and indices for vectorized offspring calculation
        K = parents
        P = parents
        phi = np.random.rand(dv,K)
        ind1 = range(0,K,2)
        ind2 = range(1,K,2)
             
        # Concatonate original parents children, and new random strings all together into new design string array
        Lam = np.hstack((Lam[:,0:P], phi[:,ind1]*Lam[:,ind1] + (1-phi[:,ind1])*Lam[:,ind2],
                      phi[:,ind2]*Lam[:,ind2] + (1-phi[:,ind2])*Lam[:,ind1], 
                    (SB[1] - SB[0])*np.random.rand(dv,S-P-K)+SB[0]));
        Lambda[i][:] = Lam
        # Update start to P such that only new string cost values are calculated
        start = P
    PI = Pi    
    return PI, Orig, Lambda
################################## Problem 2.5 (Optimize for piB using GA) ####################################################

# Call genetic algorithm function ... 
PI, Orig, Lambda = myGA(p['parents'], p['G'], p['S'], p['dv'], p['piB'],p['SB'])

# Saving your variables to a file (aka pickling) -- overwriting data every time to see how GA changes with different initial guesses
with open('GenAlg.pkl', 'wb') as f:
    pickle.dump(PI, f)
    pickle.dump(Orig, f)
    pickle.dump(Lambda, f)



plt.figure(figsize = (15,7))
plt.plot(p['x'],piBPlot(p['x']))
plt.plot(Lambda[0,0],piBPlot(Lambda[0,0]),'r*',ms = 10)
plt.plot(Lambda[:,0],piBPlot(Lambda[:,0]))
plt.plot(Lambda[-1,0],piBPlot(Lambda[-1,0]),'b*',ms = 10)
plt.title("Convergence of Genetic Algorithm in %d Generations" % p['G'])
plt.xlabel('$x$')
plt.ylabel('$\Pi_b(x)$')
plt.legend(("$\Pi_b$","$x_0$ =  %f" % Lambda[0,0],"GA Convergence",'sol = %f' % PI[0]))
plt.show()

plt.figure(figsize = (15,7))
plt.plot(range(p['G']),piBPlot(Lambda[:,0]))
plt.title("Evolution of Best Cost per Generation")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.show()

plt.figure(figsize = (15,7))
plt.plot(range(p['G']),piBPlot(np.mean(Lambda[:,0:p['parents']],1)))
plt.title("Evolution of Average Parent Cost per Generation")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.show()

plt.figure(figsize = (15,7))
plt.plot(range(p['G']),piBPlot(np.mean(Lambda,1)))
plt.title("Evolution of Average Cost per Generation")
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.show()