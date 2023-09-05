from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import random
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm

def scan_raster(T_labels, N_labels, window_dim = None):
    '''
    T_labels an array of spiketimes
    N_labels corresponding array of neuron labels
    window_dim is the size of the window to cluster the spikes
    '''
    if window_dim == None:
        window_dim = 20
        
    T_labels = np.round(T_labels).astype(int)
    T_labels, N_labels = np.unique(np.array([T_labels,N_labels]),axis=1) # This removes any spikes that occur at the same neuron at the same time
    N=max(N_labels)+1

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

    # Get the cluster assignments for each spike based on the hierarchical clustering
    clusters = np.zeros_like(T_labels)
    print('Windowing...')
    for n in tqdm(range(N)):
        idc = np.where(N_labels==n)[0]
        windows_n = windows[idc]
        if len(windows_n) > 1:
            x = np.zeros((len(windows_n),len(windows_n)))
            for i in range(windows_n.shape[0]):
                for j in range(windows_n.shape[0]):
                    common_rows = windows_n[i].intersection(windows_n[j])
                    num_identical_rows = len(common_rows)
                    x[i,j] = len(common_rows)/min(len(windows_n[i]),len(windows_n[j]))
            sim_mats.append(x)
            dissimilarity = 1 - x
            if not np.all(dissimilarity == 0):
                HDPs.append(linkage(dissimilarity, method='complete'))
                l = max(clusters)+1
                clusters[idc]= l+fcluster(linkage(dissimilarity, method='complete'), cutoff, criterion='distance')

    clusters= np.array(clusters)
    
    time_differences = []
    cluster_sq = {}
    print('Clustering...')
    
    for cluster in tqdm(np.unique(clusters)):
        temp = list(np.diff(np.unique(T_labels[clusters == cluster])))
        str_temp = str(temp)
        time_differences.append(temp)
        if str_temp in cluster_sq.keys():
            cluster_sq[str_temp] = cluster_sq[str_temp] + [cluster]
        else:
            cluster_sq[str_temp] = [cluster]
    ''' 
    This is the second round of clustering. Only patterns that repeat across multiple neurons are considered a motif. 


    with some help from chatgpt
    '''
    from collections import Counter

    # Convert the list of lists to a set of tuples to remove duplicates
    unique_sublists_set = set(tuple(sublist) for sublist in time_differences if sublist)

    # Convert the set of tuples back to a list of lists
    unique_sublists = [list(sublist) for sublist in unique_sublists_set]

    # Count the occurrences of each unique sublist in the original list
    sublist_counts = Counter(tuple(sublist) for sublist in time_differences if sublist)

    # Print the unique sublists and their respective counts
    for sublist in unique_sublists:
        count = sublist_counts[tuple(sublist)]
        print(f"{sublist}: {count} occurrences")

    sublist_keys_np = np.array([list(key) for key in sublist_counts.keys()],dtype='object')
    sublist_keys_filt = sublist_keys_np[np.array(list(sublist_counts.values())) >1] # only bother clustering repetitions that appear for more than one neuron

    ''' to visualize the clusters'''

    recovered_labels = np.zeros_like(clusters)
    for l, key in enumerate(sublist_keys_filt):
        for k in cluster_sq[str(key)]:
            recovered_labels[clusters == k] = l+1

    ''' to get the timings'''

    # Sort y according to x
    sorted_indices = np.argsort(T_labels)
    sorted_x = T_labels[sorted_indices]

    all_times = []
    all_labels = []
    for key in tqdm(sublist_keys_filt):
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
    for i in tqdm(range(len(all_times))):
        pattern = []
        pattern_template.append([])
        for time in all_times[i]:
            condition = (T_labels > time-window_dim*2) & (T_labels < time + window_dim*2)
            pattern = [tuple(k) for k in np.array([T_labels[condition]-time, N_labels[condition]]).T] # creating a list of tuples
            pattern_template[-1] += pattern # adds all points of each pattern to template_pattern
            patterns.append(pattern)
            
    if len(pattern_template) < 1:
        print("No patterns detected")
        return None, None

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
            
    from scipy.signal import correlate

    matrix_x = pattern_img
    matrix_y = pattern_img

    # Calculate cross-correlation matrix
    cross_corr_matrix = np.zeros((matrix_x.shape[0], matrix_y.shape[0]))
    
    'Comparing patterns...'

    for x_channel_idx in tqdm(range(matrix_x.shape[0])):
        for y_channel_idx in range(matrix_x.shape[0]):
            cross_corr = correlate(matrix_x[x_channel_idx,...], matrix_x[y_channel_idx,...], mode='full')
            max_corr = np.max(cross_corr)/ np.sum(matrix_x[x_channel_idx])
            cross_corr_matrix[x_channel_idx, y_channel_idx] = max_corr

    dissimilarity = cross_corr_matrix-1
    if dissimilarity.shape[0] >1:
        HDP = linkage(dissimilarity, method='complete')
    
    raster_size = (N,max(T_labels)+1)
    raster = np.zeros((raster_size))
    for (i,j) in zip(T_labels,N_labels):
        raster[j,i] =1
    
    if dissimilarity.shape[0] >1:
        method1_labels = fcluster(HDP,cutoff, criterion='distance')
    else:
        method1_labels = [1]

    pattern_convolutions = np.zeros((pattern_img.shape[0], raster.shape[1]+pattern_img.shape[2]-1))
    for j in range(pattern_img.shape[0]):
        for i in range(pattern_img.shape[1]):
            pattern_convolutions[j] += correlate(raster[i,:], pattern_img[j,i,:], mode='full')
        pattern_convolutions[j] /= np.sum(pattern_img[j,:,:]) # normalize the convolution

    detected_patterns = pattern_convolutions.copy()
    detected_patterns[detected_patterns != 1] = 0
    detected_patterns = np.sum(detected_patterns,axis=1)

    method1_pattern_winners = []
    for l in np.unique(method1_labels):
        idc = np.where(method1_labels==l)[0]
        temp = detected_patterns[method1_labels==l]
        method1_pattern_winners.append(idc[temp == max(temp)][0])

    method1_pattern_template = pattern_img[method1_pattern_winners]
    method1_pattern_template.shape
    print('Method 1 detected patterns:', method1_pattern_winners)

    T=max(T_labels)
    M = len(method1_pattern_winners)
    D = method1_pattern_template.shape[2]

    
    'Checking patterns...'
    sanity_check = np.zeros((T,M))
    for j in tqdm(range(M)):
        for i in range(T):
            if raster[:,i:i+D].shape[1] == D:
                sanity_check[i,j] = np.sum(method1_pattern_template[j,...]*raster[:,i:i+D])
        sanity_check[:,j] = sanity_check[:,j]/np.max(sanity_check[:,j])
        
    return method1_pattern_template, sanity_check





















    