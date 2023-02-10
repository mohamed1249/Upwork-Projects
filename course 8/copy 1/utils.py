import numpy as np
import pandas as pd


def load_networkA():
    param = {'rho': 1000, 'g': 9.81, 'nu': 1.004 * 1e-6, 'eta': 0.85}

    # pipe data ....................................
    pipes = pd.DataFrame(np.array([
        [1, 0, 1, 610 * 1e-3, 0.381 * 1e-3, 580],  # Pipe 1
        [2, 1, 2, 460 * 1e-3, 0.381 * 1e-3, 610],  # Pipe 2
        [3, 3, 2, 610 * 1e-3, 0.381 * 1e-3, 610],  # Pipe 3
        [4, 0, 3, 460 * 1e-3, 0.381 * 1e-3, 425],  # Pipe 4
        [5, 1, 4, 610 * 1e-3, 0.381 * 1e-3, 460],  # Pipe 5
        [6, 6, 4, 610 * 1e-3, 0.381 * 1e-3, 360],  # Pipe 6
        [7, 4, 5, 460 * 1e-3, 0.381 * 1e-3, 610],  # Pipe 7
        [8, 6, 2, 380 * 1e-3, 0.381 * 1e-3, 1000],  # Pipe 8
        [9, 7, 7, 460 * 1e-3, 0.381 * 1e-3, 400],  # Pipe 9
        [10, 7, 5, 300 * 1e-3, 0.254 * 1e-3, 500],  # Pipe 10
        [11, 7, 8, 460 * 1e-3, 0.229 * 1e-3, 200],  # Pipe 11
        [12, 5, 8, 610 * 1e-3, 0.381 * 1e-3, 460],  # Pipe 12
        [13, 2, 5, 610 * 1e-3, 0.381 * 1e-3, 460]]),  # Pipe 13
        columns=['id', 'start', 'end', 'D', 'eps', 'L'])

    pipes['k'] = 8 * pipes['L'] / (param['g'] * (np.pi ** 2) * (pipes['D'] ** 5))
    numpipes = len(pipes)

    # initial guess
    q0 = np.zeros(numpipes)
    q0[0] = 0
    q0[1] = 0
    q0[2] = 0.315
    q0[3] = 0.473
    q0[4] = 0
    q0[5] = 0
    q0[6] = 0.158
    q0[7] = 0.631
    q0[8] = 0.158
    q0[9] = 0.158
    q0[10] = 0
    q0[11] = 0.473
    q0[12] = 0.315

    pipes['Q0'] = q0

    # loops ........................
    nloops = 5
    loops = [[] for _ in range(nloops)]
    loops[0] = [1, 2, -3, -4]  # list of pipe ids
    loops[1] = [5, 7, -13, -2]
    loops[2] = [-6, 9, 10, -7]
    loops[3] = [-10, 11, -12]
    loops[4] = [5, -6, 8, -2]

    # create loops matrix
    loopsind = np.zeros((numpipes, nloops))
    for loopind, loop in enumerate(loops):
        for pipeid in loop:
            loopsind[abs(pipeid) - 1, loopind] = np.sign(pipeid)

    return (pipes, loopsind, param)




def load_networkB():
    
    params = {'rho': 1000, 'g': 9.81, 'nu': 1.004 * 1e-6, 'eta': 0.85}

    # pipe data ....................................
    pipes = pd.DataFrame(np.array([
        [1,  0, 1, 600 * 1e-3, 0.4 * 1e-3,  580],   # Pipe 1
        [2,  1, 2, 450 * 1e-3, 0.4 * 1e-3,  600],   # Pipe 2
        [3,  3, 2, 600 * 1e-3, 0.4 * 1e-3,  600],   # Pipe 3
        [4,  0, 3, 450 * 1e-3, 0.4 * 1e-3,  430],   # Pipe 4
        [5,  1, 4, 600 * 1e-3, 0.4 * 1e-3,  450],   # Pipe 5
        [6,  6, 4, 600 * 1e-3, 0.4 * 1e-3,  370],   # Pipe 6
        [7,  4, 5, 450 * 1e-3, 0.4 * 1e-3,  600],   # Pipe 7
        [8,  6, 7, 450 * 1e-3, 0.4 * 1e-3,  400],   # Pipe 8
        [9,  7, 5, 399 * 1e-3, 0.4 * 1e-3,  500],   # Pipe 9
        [10, 2, 5, 600 * 1e-3, 0.3 * 1e-3,  450],   # Pipe 10
        [11, 6, 2, 400 * 1e-3, 0.28 * 1e-3,1000],   # Pipe 11
        [12, 4, 3, 400 * 1e-3, 0.4 * 1e-3,  900]]),  # Pipe 12
        columns=['id', 'start', 'end', 'D', 'eps', 'L'])

    pipes['k'] = 8 * pipes['L'] / (params['g'] * (np.pi ** 2) * (pipes['D'] ** 5))
    numpipes = len(pipes)

    # initial guess
    q0 = np.zeros(numpipes)
    q0[0] = 0.2   # Pipe 1
    q0[1] = 0.2   # Pipe 2
    q0[2] = 0.2   # Pipe 3
    q0[3] = 0.2   # Pipe 4
    q0[4] = 0.0   # Pipe 5
    q0[5] = 0.0   # Pipe 6
    q0[6] = 0.12  # Pipe 7
    q0[7] = 0.6   # Pipe 8
    q0[8] = 0.0   # Pipe 9
    q0[9] = 0.0   # Pipe 10
    q0[10] = 0.2   # Pipe 11
    q0[11] = 0.2   # Pipe 12

    pipes['Q0'] = q0

    # loops ........................
    nloops = 5
    loops = [[] for _ in range(nloops)]
    loops[0] = [-4,1,2,-3]  # list of pipe ids
    loops[1] = [-2,5,7,-10]
    loops[2] = [-7,-6,8,9]
    loops[3] = [-4,1,5,12]
    loops[4] = [-2,5,-6,11]

    # create loops matrix
    loopsind = np.zeros((numpipes, nloops))
    for loopind, loop in enumerate(loops):
        for pipeid in loop:
            loopsind[abs(pipeid) - 1, loopind] = np.sign(pipeid)
            
            
    Qsd = {
        0 :  0.4,
        1 :  0.0,
        2 : -0.6,
        3 : -0.2,
        4 :  0.32,
        5 : -0.12,
        6 :  0.8,
        7 : -0.6,}
    
    node_ids = set()
    node_ids.update(pipes['start'].values)
    node_ids.update(pipes['end'].values)
    numnodes = len(node_ids)
    numpipes = len(pipes)


    return pipes, loopsind, params, Qsd, numnodes, numpipes

def fhalland(Q, pipes, params):
    nu = params['nu']
    g = params['g']
    D = pipes['D'].values
    eps = pipes['eps'].values
    Re = 4. * abs(Q) / D / nu / np.pi
    eoverD = eps / D
    return (-1.8 * np.log10((eoverD / 3.7) ** 1.11 + 6.9 / Re)) ** (-2)

def head_loss(Q, pipes, params):
    f = fhalland(Q, pipes, params)
    D = pipes['D'].values
    L = pipes['L'].values
    V = 4 * Q / (D ** 2) / np.pi
    Hl = (f * (L / D) * (V ** 2)) / 2 / params['g']
    return Hl

def run_hardy_cross(Q, pipes, loopsind, params, dorecord):
    
    # in case its a pandas time series
    D = pipes['D'].values
    L = pipes['L'].values
    eps = pipes['eps'].values

    if dorecord:
        record = {
            'HloverQ': np.empty(0),
            'sumHlQ': np.empty(0),
            'sumHl': np.empty(0),
            'sumdQ': np.empty(0),
            'dQ': np.empty(0),
            'Q': np.empty(0)}

    alpha = 0.5
    maxiterations = 100

    dQ = Q
    c = 0
    exeeded_max_iter = False
    while np.linalg.norm(dQ, 2) >= 0.001:
        c = c + 1

        # 1)
        Hl = head_loss(Q, pipes, params)

        # 2)
        HloverQ = abs(Hl / Q)
        HloverQ[Q == 0] = 0
        sumHlQ = HloverQ @ abs(loopsind)
        sumHl = (np.sign(Q) * abs(Hl)) @ loopsind

        # 3)
        sumdQ = -alpha * sumHl / sumHlQ
        dQ = sumdQ @ loopsind.T
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

    Hl = head_loss(Q, pipes, params)

    return (Q, Hl)


def calculate_pressures(Q, Hl, params, numnodes):

    P = np.zeros(numnodes)
    P[0] = 100000 / (params['rho'] * params['g'])  # [meters of water]
    P[1] = P[0] - np.sign(Q[0]) * np.abs(Hl[0])    # 2 <-1-  1
    P[2] = P[1] - np.sign(Q[1]) * np.abs(Hl[1])    # 3 <-2-  2
    P[3] = P[0] - np.sign(Q[3]) * np.abs(Hl[3])    # 4 <-4-  1
    P[4] = P[1] - np.sign(Q[4]) * np.abs(Hl[4])    # 5 <-5-  2
    P[5] = P[4] - np.sign(Q[6]) * np.abs(Hl[6])    # 6 <-7-  5
    P[6] = P[4] + np.sign(Q[5]) * np.abs(Hl[5])    # 7  -6-> 5
    P[7] = P[6] - np.sign(Q[7]) * np.abs(Hl[7])    # 8 <-8-> 7
    return P

def calculate_pump(P, Qsd, params, printit=False):
    P_Pa = P*params['rho'] * params['g']

    power1 = P_Pa[0] * Qsd[0] / params['eta'] 
    power5 = P_Pa[4] * Qsd[4] / params['eta'] 
    power7 = P_Pa[6] * Qsd[6] / params['eta'] 

    e1 = 24 * power1 / 1000  # kwh
    e5 = 24 * power5 / 1000  # kwh
    e7 = 24 * power7 / 1000  # kwh
    
    if printit:
        print(f'Node 1: {e1} KWh')
        print(f'Node 5: {e5} KWh')
        print(f'Node 7: {e7} KWh')
            
    return e1 + e5 + e7  
