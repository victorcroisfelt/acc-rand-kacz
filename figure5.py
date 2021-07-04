########################################
#   figure5.py
#
#   Description. Script used to generate Figure 5 of the paper.
#
#   Author. @victorcroisfelt
#
#   Date. May 21, 2021
#
#   This code is part of the code package used to generate the results of the
#   paper:
#
#   V. C. Rodrigues, A. Amiri, T. Abrao, E. D. Carvalho and P. Popovski,
#   "Accelerated Randomized Methods for Receiver Design in Extra-Large Scale
#   MIMO Arrays," in IEEE Transactions on Vehicular Technology,
#   doi:  10.1109/TVT.2021.3082520.
#
#   Available on: https://ieeexplore.ieee.org/document/9437708
########################################

########################################
# Preamble
########################################

import numpy as np
import time
from datetime import datetime

import multiprocessing
from joblib import Parallel
from joblib import dump, load

from newfunctions import *
from commsetup import *

import matplotlib.pyplot as plt

# Obtain the number of processors
num_cores = multiprocessing.cpu_count()

# Random seed
np.random.seed(42)

# Treating errors in numpy
np.seterr(divide='raise', invalid='raise')

########################################
# System parameters
########################################

# Number of antennas
M = 256

# Number of users
K = 32

# Number of effective antennas
D = 8

########################################
# Environment parameters
########################################

# Define pre-processing SNR
SNRdB = 0
SNR = 10**(SNRdB/10)

########################################
# Simulation parameters
########################################

# Define number of simulation setups
nsetups = 10

# Define number of channel realizations
nchnlreal = 100

########################################
# Running simulation
########################################

# Simulation header
print('--------------------------------------------------')
now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('XL-MIMO: BER vs niter')
print('\t M = '+str(M))
print('\t K = '+str(K))
print('\t D = '+str(D))
print('\t SNR = '+str(SNRdB))
print('--------------------------------------------------')

# Prepare to save simulation results
ber_mr = np.zeros((nsetups, nchnlreal), dtype=np.double)
ber_rzf = np.zeros((nsetups, nchnlreal), dtype=np.double)

ber_nrk = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_rk  = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_grk = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_rsk = np.zeros((niter_range.size, nsetups, nchnlreal), dtype=np.double)

# Obtain qam transmitted signals
tx_symbs, x_ = qam_transmitted_signals(K, nsetups)

# Go through all setups
for s in range(nsetups):

    print(f"setup: {s}/{nsetups-1}")

    timer_setup = time.time()

    # Generate communication setup
    H = extra_large_mimo(M, K, D, nchnlreal)

    # Compute the Gramian matrix
    G = channel_gramian_matrix(H)

    # Compute received signal
    y_ = received_signal(SNR, x_[s], H)

    # Perform MR receive combining
    xhat_soft_mr = mrc_detection(H, y_)

    # Evaluate RZF performance
    ber_mr[s] = ber_evaluation(xhat_soft_mr, tx_symbs[s])

    # Perform RZF receive combining
    xhat_soft_rzf, xhat_rzf, Dinv_rzf = rzf_detection(SNR, H, G, y_)

    # Evaluate RZF performance
    ber_rzf[s] = ber_evaluation(xhat_soft_rzf, tx_symbs[s])

    # Perform RK-based RZF schemes
    with Parallel(n_jobs=num_cores) as parl:
        xhat_soft_nrk, xhat_soft_rk, xhat_soft_grk, xhat_soft_rsk = kaczmarz_detection(SNR, H, G, y_, Dinv_rzf, niter_range, parl=parl)

    # Go through each iteration point
    for niter in range(len(niter_range)):
        ber_nrk[niter, s] = ber_evaluation(xhat_soft_nrk[niter], tx_symbs[s])
        ber_rk[niter, s]  = ber_evaluation(xhat_soft_rk[niter], tx_symbs[s])
        ber_grk[niter, s] = ber_evaluation(xhat_soft_grk[niter], tx_symbs[s])
        ber_rsk[niter, s] = ber_evaluation(xhat_soft_rsk[niter], tx_symbs[s])

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('xlmimo_ber_vs_niter_K'+str(K)+'_D'+str(D)+'.npz',
    M=M,
    K=K,
    D=D,
    niter_range=niter_range,
    ber_mr=ber_mr,
    ber_rzf=ber_rzf,
    ber_nrk=ber_nrk,
    ber_rk=ber_rk,
    ber_grk=ber_grk,
    ber_rsk=ber_rsk)

# Compute average values
ber_mr_avg  = (ber_mr.mean(axis=-1)).mean(axis=-1)
ber_rzf_avg = (ber_rzf.mean(axis=-1)).mean(axis=-1)
ber_nrk_avg = (ber_nrk.mean(axis=-1)).mean(axis=-1)
ber_rk_avg  = (ber_rk.mean(axis=-1)).mean(axis=-1)
ber_grk_avg = (ber_grk.mean(axis=-1)).mean(axis=-1)
ber_rsk_avg = (ber_rsk.mean(axis=-1)).mean(axis=-1)

fig, ax = plt.subplots()

ax.plot(niter_range, ber_mr_avg*np.ones((len(niter_range))))
ax.plot(niter_range, ber_rzf_avg*np.ones((len(niter_range))))
ax.plot(niter_range, ber_nrk_avg)
ax.plot(niter_range, ber_rk_avg)
ax.plot(niter_range, ber_grk_avg)
ax.plot(niter_range, ber_rsk_avg)

ax.set_yscale('log')

plt.show()
