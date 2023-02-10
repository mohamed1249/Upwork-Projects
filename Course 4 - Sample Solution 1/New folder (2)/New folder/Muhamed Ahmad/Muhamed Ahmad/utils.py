import numpy as np
import pandas as pd

def load_stuff():
    param = {'rho' : 1000, 'g' : 9.81, 'nu' : 1.004 * 1e-6}
    
    # Pipes data ---------------------------------------------------
    pipes = pd.DataFrame(np.array([
        [1, 0, 1, 450e-3, 0.26e-3, 730 ],  #Pipe 1
        [2, 1, 2, 350e-3, 0.26e-3, 600 ],  #Pipe 2
        [3, 0, 2, 450e-3, 0.26e-3, 730 ],  #Pipe 3
        [4, 1, 3, 300e-3, 0.26e-3, 550 ],  #Pipe 4
        [5, 2, 3, 300e-3, 0.26e-3, 580 ],  #Pipe 5
        [6, 4, 3, 250e-3, 0.23e-3, 400 ],  #Pipe 6
        [7, 3, 5, 400e-3, 0.26e-3, 570 ],  #Pipe 7
        [8, 2, 5, 450e-3, 0.23e-3, 600 ],  #Pipe 8
        [9, 4, 5, 350e-3, 0.26e-3, 520 ],  #Pipe 9
        [10, 4, 6, 600e-3, 0.26e-3, 360],  #Pipe 10
        [11, 6, 5, 380e-3, 0.26e-3, 520],  #Pipe 11
        [12, 5, 7, 300e-3, 0.26e-3, 320],  #Pipe 12
        [13, 6, 7, 350e-3, 0.26e-3, 180]]),  #Pipe 13
        columns=['id', 'start', 'end', 'D', 'eps', 'L'])
    
    pipes['k'] = 8 * pipes['L'] / (param['g'] * (np.pi ** 2) * (pipes['D'] ** 5))
    numpipes = len(pipes)
    numnodes = 8
    
    #Loops --------------------------------------------------------
    nloops = 6
    loops = [[] for _ in range(nloops)]
    loops[0] = [1, 2, -3]   #List of pipe ids
    loops[1] = [4, -5, -2]
    loops[2] = [5, 7, -8]
    loops[3] = [-7, -6, 9]
    loops[4] = [-9, 10, 11]
    loops[5] = [-11, 13, -12]
    
    #create loops matrix
    loopsind = np.zeros((numpipes, nloops))
    for loopind, loop in enumerate(loops):
        for pipeid in loop:
            loopsind[abs(pipeid) - 1, loopind] = np.sign(pipeid)
    loopsind = loopsind.T
    
    Qsd = np.zeros(numnodes)
    Qsd[0] = 0.9
    Qsd[1] = -0.1
    Qsd[2] = -0.28
    Qsd[3] = -0.16
    Qsd[4] = 0.5
    Qsd[5] = -0.25
    Qsd[6] = -0.32
    Qsd[7] = -0.29
    
    return pipes, loopsind, param, Qsd, numpipes, numnodes, nloops

def fhalland(Q, pipes, param):
    
    D = pipes['D'].values
    eps = pipes['eps'].values
    nu = param['nu']
    
    Re = 4. * abs(Q) / D / nu / np.pi
    eoverD = eps / D
    return (-1.8 * np.log10((eoverD / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)

def head_loss(Q, pipes, param):
    
    D = pipes['D'].values
    eps = pipes['eps'].values
    L = pipes['L'].values
    
    f = fhalland(Q, pipes, param)
    V = 4 * Q / (D ** 2) / np.pi
    Hl = (f * (L / D) * (V ** 2)) / 2 / param['g']
    return Hl

def run_hardy_cross(Q, pipes, loopsind, param, dorecord):
    
    D = pipes['D'].values
    eps = pipes['eps'].values
    L = pipes['L'].values
    
    
    #in case its a pandas time series
    D = np.array(D)
    L = np.array(L)
    eps = np.array(eps)
    
    if dorecord:
        record = {
            'HloverQ' : np.empty(0),
            'sumHlQ' : np.empty(0),
            'sumHl' : np.empty(0),
            'sumdQ' : np.empty(0),
            'dQ' : np.empty(0),
            'Q' : np.empty(0)}
    
    alpha = 0.5
    maxiterations = 100
    
    dQ = Q
    C = 0
    exeeded_max_iter = False
    while np.linalg.norm(dQ, 2) >= 0.001:
        c = c + 1
        
        # 1)
        Hl = head_loss(Q, pipes, param)
        
        # 2)
        HloverQ = abs(Hl / Q)
        HloverQ[Q == 0] = 0
        sumHlQ = HloverQ @ abs(loopsind.T)
        sumHl = (np.sign(Q) * abs(Hl)) @ loopsind.T
        
        #3)
        sumdQ = -alpha * sumHl / sumHlQ
        dQ = sumdQ @ loopsind
        Q = Q + dQ
        
        if dorecord:
            record.HloverQ.append(HloverQ)
            record.sumHlQ.append(sumHlQ)
            record.sumHl.append(sumHl)
            record.sumdQ.append(sumdQ)
            record.dQ.append(dQ)
            record.Q.append(Q)
            
        if c > maxiterations:
            exeeded_max_iter = True
            break
            
    if exeeded_max_iter:
        return ([], [])
    
    Hl = head_loss(Q, pipes, param)
    
    return (Q,Hl)

def calculate_pressure(Q, Hl):
    P = np.zeros(9)
        
    