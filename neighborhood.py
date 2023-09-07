import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_60(T_labels, N_labels, window_dim = None, alpha = 0.01):
    N=max(N_labels)+1
    if window_dim == None:
        window_dim = 50

    windows = np.zeros((len(T_labels)),dtype='object')
    for i,window_time in enumerate(T_labels):
        condition = (T_labels > window_time-window_dim) & (T_labels < window_time + window_dim)
        window = np.array([T_labels[condition]-window_time, N_labels[condition]]).T
        window =  {tuple(row) for row in  window}
        windows[i] = window
    
    neighborhoods = np.zeros(N,dtype='object')
    for n in tqdm(range(N)):
        windows_n = windows[N_labels==n]
        all_points_n = []
        for i,w in enumerate(windows_n):
            w.remove((0,n))
            for j in range(len(windows_n)):
                if i!= j:
                    all_points_n.append(windows_n[i].intersection(windows_n[j]))
        all_points_n = [k for k in all_points_n if k!= set()]
        x = np.array([])
        y = np.array([])
        for points in all_points_n:
            x = np.hstack((x,np.array([k for k in points]).T[0]))
            y = np.hstack((y,np.array([k for k in points]).T[1]))
        neighborhoods[n] = np.array((x,y))
        if n >=60:
            break
        
    plots = np.zeros(8**2,dtype='object')
    plots[1:7] = np.arange(1,61)[:6]
    plots[8:64-8] = np.arange(1,61)[6:60-6]
    plots[64-8+1:64-8+1+6] = np.arange(1,61)[60-6:60+1]
    plots -=1
    
    fig, axs = plt.subplots(8, 8,  figsize=(16, 16))
    i=0
    for i,ax in enumerate(axs.flat):
        if plots[i] >=0:
            ax.scatter(*neighborhoods[plots[i]],c='black',s=1,alpha=alpha)
            ax.set_title(f'{plots[i]+1}')
        ax.axis('off')
    fig.suptitle('Averaged spikes per neuron',fontsize=24)
    print('Plotting...')
    plt.show()
    
def plot_random(T_labels, N_labels, window_dim = None, alpha = 0.01):
    N=max(N_labels)+1
    if window_dim == None:
        window_dim = 50
    windows = np.zeros((len(T_labels)),dtype='object')
    for i,window_time in enumerate(T_labels):
        condition = (T_labels > window_time-window_dim) & (T_labels < window_time + window_dim)
        window = np.array([T_labels[condition]-window_time, N_labels[condition]]).T
        window =  {tuple(row) for row in  window}
        windows[i] = window
    
    neighborhoods = np.zeros(N,dtype='object')
    for n in tqdm(range(N)):
        windows_n = windows[N_labels==n]
        all_points_n = []
        for i,w in enumerate(windows_n):
            w.remove((0,n))
            for j in range(len(windows_n)):
                if i!= j:
                    all_points_n.append(windows_n[i].intersection(windows_n[j]))
        all_points_n = [k for k in all_points_n if k!= set()]
        x = np.array([])
        y = np.array([])
        for points in all_points_n:
            x = np.hstack((x,np.array([k for k in points]).T[0]))
            y = np.hstack((y,np.array([k for k in points]).T[1]))
        neighborhoods[n] = np.array((x,y))
        if n >=60:
            break
        
    dim = 0
    condition= True
    while condition:
        dim+=1
        if dim**2 > N:
            condition = False
            
    plots = np.arange(N)
    plots = np.hstack((plots, np.array(int(dim**2 - N)*[-1])))
    
    fig, axs = plt.subplots(dim, dim,  figsize=(16, 16))
    i=0
    for i,ax in enumerate(axs.flat):
        if plots[i] >=0:
            ax.scatter(*neighborhoods[plots[i]],c='black',s=1,alpha=alpha)
            ax.set_title(f'{plots[i]+1}')
        ax.axis('off')
    fig.suptitle('Averaged spikes per neuron',fontsize=24)
    print('Plotting...')
    plt.show()