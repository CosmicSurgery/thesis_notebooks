from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np
from scipy.signal import correlate
from collections import Counter
from tqdm import tqdm


def scan_raster(T_labels, N_labels, window_dim = None):
    '''
    T_labels an array of spiketimes
    N_labels corresponding array of neuron labels
    window_dim is the size of the window to cluster the spikes
    '''
    if window_dim == None:
        window_dim = 100
        
    T_labels = np.round(T_labels).astype(int)
    T_labels, N_labels = np.unique(np.array([T_labels,N_labels]),axis=1) # This removes any spikes that occur at the same neuron at the same time
    N=max(N_labels)+1

    windows = np.zeros((len(T_labels)),dtype='object')
    for i,window_time in enumerate(T_labels):
        condition = (T_labels > window_time-window_dim) & (T_labels < window_time + window_dim)
        window = np.array([T_labels[condition]-window_time, N_labels[condition]]).T
        window =  {tuple(row) for row in  window}
        windows[i] = window

        
    # Set the cutoff value for clustering
    cutoff = 0
    lr = 0.01

    max_iter=50
    lr = 0.01
    iter_ = 0

    opt_cutoff = cutoff
    max_seq_rep = 0

    while iter_ <= max_iter: # this is just a for loop...

        clusters = cluster_windows(cutoff)
        cluster_sq, sq_counts, sublist_keys_filt = check_seq(clusters, cutoff)

        if len(sublist_keys_filt) != 0:
            max_ = np.max([len(k) for k in sublist_keys_filt])
            if max_seq_rep < max_:
                max_seq_rep = max_
                opt_cutoff=cutoff

        cutoff += lr
        iter_ +=1


        print(f'iter - {iter_/max_iter} | cutoff - {cutoff} | opt_cutoff - {opt_cutoff} | most_detections - {max_seq_rep}',end='\r')

    clusters = _cluster_windows(cutoff)
    cluster_sq, sq_counts, sublist_keys_filt = _check_seq(clusters, cutoff)
        

    ''' to get the timings'''

    # Sort y according to x
    sorted_indices = np.argsort(T_labels)
    sorted_x = T_labels[sorted_indices]

    all_times = []
    all_labels = []
    for key in sublist_keys_filt:
        pattern_repetition_labels = np.zeros((len(cluster_sq[str(key)]),len(clusters)))
        for i,k in enumerate(cluster_sq[str(key)]):
            pattern_repetition_labels[i][clusters==k] = 1
            pattern_repetition_labels[i] *= np.cumsum(pattern_repetition_labels[i])
        pattern_repetition_labels = np.sum(pattern_repetition_labels,axis=0,dtype='int')
        all_labels.append(pattern_repetition_labels)

        sorted_y = pattern_repetition_labels[sorted_indices]
        pattern_times = np.array([sorted_x[sorted_y==i][0] for i in range(1,max(pattern_repetition_labels)+1)])
        all_times.append(pattern_times)

    pattern_template = []
    patterns = []
    for i in range(len(all_times)):
        pattern = []
        pattern_template.append([])
        for time in all_times[i]:
            condition = (T_labels > time-window_dim*2) & (T_labels < time + window_dim*2)
            pattern = [tuple(k) for k in np.array([T_labels[condition]-time, N_labels[condition]]).T] # creating a list of tuples
            pattern_template[-1] += pattern # adds all points of each pattern to template_pattern
            patterns.append(pattern)

    for i,pattern in enumerate(pattern_template):
        counts = [pattern.count(k) for k in pattern]
        pattern_template[i] = np.array(pattern)[np.where(counts == np.max(counts))[0]]
        pattern_template[i][:,0] -= min(pattern_template[i][:,0])
        pattern_template[i] = np.unique(pattern_template[i],axis=0)
    
    win_size = (N,1+max([max(k[:,0]) for k in pattern_template]))
    pattern_img = np.zeros((len(pattern_template),*win_size))
    for p,pattern in enumerate(pattern_template):
        for (i,j) in pattern:
            pattern_img[p,j,i] = 1

    return pattern_template, sublist_keys_filt

def _cluster_windows(cutoff):
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
  
    
    return clusters

def _check_seq(clusters, cutoff):

    time_differences = []
    cluster_sq = {}
    for cluster in np.unique(clusters):
        temp = list(np.diff(np.unique(T_labels[clusters == cluster])))
        str_temp = str(temp)
        time_differences.append(temp)
        if str_temp in cluster_sq.keys():
            cluster_sq[str_temp] = cluster_sq[str_temp] + [cluster]
        else:
            cluster_sq[str_temp] = [cluster]

    # Convert the list of lists to a set of tuples to remove duplicates
    unique_sublists_set = set(tuple(sublist) for sublist in time_differences if sublist)

    # Convert the set of tuples back to a list of lists
    unique_sublists = [list(sublist) for sublist in unique_sublists_set]

    # Count the occurrences of each unique sublist in the original list
    sublist_counts = Counter(tuple(sublist) for sublist in time_differences if sublist)

    # Print the unique sublists and their respective counts
    sq_counts = np.zeros(len(sublist_counts)) 
    for i,sublist in enumerate(unique_sublists):
        count = sublist_counts[tuple(sublist)]
        sq_counts[i] = count
    #     print(f"{sublist}: {count} occurrences")
    sublist_keys_np = np.array([list(key) for key in sublist_counts.keys()],dtype='object')
    sublist_keys_filt = sublist_keys_np[np.array(list(sublist_counts.values())) >1] # only bother clustering repetitions that appear for more than one neuron
    
    return cluster_sq, sq_counts, sublist_keys_filt


















    