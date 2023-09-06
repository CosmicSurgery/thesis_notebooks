from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

def cluster_plots(T_labels, N_labels, window_dim = None):
    if window_dim == None:
        window_dim = 50
        
    T_labels = np.round(T_labels).astype(int)
    T_labels, N_labels = np.unique(np.array([T_labels,N_labels]),axis=1) # This removes any spikes that occur at the same neuron at the same time
    N= int(max(N_labels)+1)
    
    windows = np.zeros((len(T_labels)),dtype='object')
    for i,window_time in enumerate(T_labels):
        condition = (T_labels > window_time-window_dim) & (T_labels < window_time + window_dim)
        window = np.array([T_labels[condition]-window_time, N_labels[condition]]).T
        window =  {tuple(row) for row in  window}
        windows[i] = window

    HDPs = []
    sim_mats = []
    # Set the cutoff value for clustering
    cutoff = 1
    
    HDPs = []
    sim_mats = []
    
    # Get the cluster assignments for each spike based on hierarchical clustering
    clusters = np.zeros_like(T_labels)
    for n in range(N):
        idc = np.where(N_labels==n)[0]
        windows_n = windows[idc]
        if len(windows_n) > 1:
            x = np.zeros((len(windows_n),len(windows_n)))
            for i in range(windows_n.shape[0]):
                for j in range(windows_n.shape[0]):
                    common_rows = windows_n[i].intersection(windows_n[j])
                    num_identical_rows = len(common_rows)
                    x[i,j] = len(common_rows)/min(len(windows_n[i]),len(windows_n[j]))
            np.fill_diagonal(x,0)# make sure the diagonals are zero, this is important the more spikes there are...
            sim_mats.append(x) 
            dissimilarity = x-1
            if not np.all(dissimilarity == 0):
                HDPs.append(linkage(dissimilarity, method='complete'))
                l = max(clusters)+1
                clusters[idc]= l+fcluster(linkage(dissimilarity, method='complete'), cutoff, criterion='distance')
                fig = plt.figure(figsize=(25, 6))
                plt.imshow(dissimilarity)

                fig = plt.figure(figsize=(25, 6))
                HDP = linkage(dissimilarity, method='complete')
                dn = dendrogram(HDP)

                # Add a horizontal line at the cutoff value of 1
                plt.axhline(y=cutoff, color='r', linestyle='--', label=f'Cutoff at {cutoff}')

                # Add labels and legend
                plt.title(f"neuron {n}", fontsize=24)
                plt.ylabel('Distance', fontsize=24)
                plt.tick_params(labelsize=20)
                plt.legend(fontsize=24)
                plt.show()
        
     