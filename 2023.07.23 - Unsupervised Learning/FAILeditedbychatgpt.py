import numpy as np
import matplotlib.pyplot as plt

def generate_K(M, N, D, nrn_fr):
    K_dense = np.random.randint(0, 999, (N, D, M))
    K_dense[K_dense < nrn_fr] = 1
    K_dense[K_dense >= nrn_fr] = 0
    K_sparse = np.where(K_dense)
    return K_dense, K_sparse

def generate_B(M, T, pg_fr):
    B_dense = np.random.randint(0, 999, (M, T))
    B_dense[B_dense < pg_fr] = 1
    B_dense[B_dense >= pg_fr] = 0
    B_sparse = np.where(B_dense)
    return B_dense, B_sparse

def generate_A(K_dense, B_dense, background_noise_fr):
    N, T, M = K_dense.shape[0], B_dense.shape[1], K_dense.shape[2]
    A_dense = np.zeros((N, T + D, M + 1))
    A_dense[..., -1] = np.random.randint(0, 999, (N, T + D))
    A_dense[..., -1] = (A_dense[..., -1] < background_noise_fr).astype('int')
    B_sparse = np.where(B_dense)
    for i in range(len(B_sparse[0])):
        t, b = B_sparse[1][i], B_sparse[0][i]
        A_dense[:, t:t + D, b] += K_dense[..., b]
    A_dense[A_dense > 1] = 1
    A_sparse = np.where(A_dense[..., :-1])
    return A_dense, A_sparse

# Parameters
M = 4  # Number of Spiking motifs
N = 10  # Number of input neurons
D = 31  # temporal depth of receptive field
T = 1000
dt = 1
nrn_fr = 40  # hz
pg_fr = 4  # hz
background_noise_fr = 0  # h

# Generate K_dense, K_sparse, B_dense, B_sparse, A_dense, and A_sparse
np.random.seed(41)
K_dense, K_sparse = generate_K(M, N, D, nrn_fr)
B_dense, B_sparse = generate_B(M, T, pg_fr)

# Generate A_dense and A_sparse
A_dense = np.zeros((N, T + D, M))
A_dense[..., -1] = np.random.randint(0, 999, (N, T + D))
A_dense[..., -1] = (A_dense[..., -1] < background_noise_fr).astype('int')
B_sparse = np.where(B_dense)
for i in range(len(B_sparse[0])):
    t, b = B_sparse[1][i], B_sparse[0][i]
    A_dense[:, t:t + D, b] += K_dense[..., b]

A_sparse = np.where(A_dense)

# Convert A_dense to binary (0 or 1) representation
A_dense[A_dense > 1] = 1

# Compute the convolution of K_dense with A_dense
test = np.zeros((N, T + D, M))
for m in range(M):
    test[:, :, m] = np.sum(K_dense[..., m, None] * A_dense[:, None, :, m], axis=2)


# ... (rest of the code remains the same as before)


# for matplotlib
colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
disp_figs=False
if disp_figs:
    fig,ax = plt.subplot_mosaic('AB;CD',figsize=(12,6))
    [ax[k].scatter(K_sparse[1][K_sparse[2]==i], K_sparse[0][K_sparse[2]==i],c=colors[i],s=10) for i,k in enumerate(['A','B','C','D'])]
    plt.show()

plt.figure(figsize=(12,3))
plt.scatter(B_sparse[1],B_sparse[0],c=colors[B_sparse[0]],s=100)
plt.show()

plt.figure(figsize=(12,3))
plt.scatter(A_sparse[1],A_sparse[0],c=colors[A_sparse[2]],alpha=0.9,s=100,marker='.')
plt.show()

# Sanity Check
test = np.sum(K_dense[..., None] * A_dense[:, :, None], axis=0)
test /= np.max(test, axis=0)

plt.figure(figsize=(12, 3))
for i in range(M):
    plt.plot(test[:, i], color=colors[i], alpha=0.5)

plt.show()
