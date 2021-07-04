########################################
#   figure3.py
#
#   Description. Script used to generate Figure 3 of the paper.
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

import palettable

# Obtain color vector
colors = palettable.colorbrewer.qualitative.Set2_7.mpl_colors

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
M = 64

# Number of users
K = 8

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

# Define range of the number of iterations
niter_range = np.arange(1, 257)

########################################
# Running simulation
########################################

# Simulation header
print('--------------------------------------------------')
now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('M-MIMO: BER vs niter')
print('\t M = '+str(M))
print('\t K = '+str(K))
print('\t SNR = '+str(SNRdB))
print('--------------------------------------------------')

# Prepare to save simulation results
ber_mr = np.zeros((2, nsetups, nchnlreal), dtype=np.double)
ber_rzf = np.zeros((2, nsetups, nchnlreal), dtype=np.double)

ber_nrk = np.zeros((2, niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_rk  = np.zeros((2, niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_grk = np.zeros((2, niter_range.size, nsetups, nchnlreal), dtype=np.double)
ber_rsk = np.zeros((2, niter_range.size, nsetups, nchnlreal), dtype=np.double)

# Obtain qam transmitted signals
tx_symbs, x_ = qam_transmitted_signals(K, nsetups)

# Go through all setups
for s in range(nsetups):

    print(f"setup: {s}/{nsetups-1}")

    timer_setup = time.time()

    # Generate communication setup
    Huncorr, Hcorr = massive_mimo(M, K, nchnlreal, iota=.5)

    for case in range(2):

        # Decide the current correlation model
        if case == 0:
            H = Huncorr.copy()
        else:
            H = Hcorr.copy()

        # Compute the Gramian matrix
        G = channel_gramian_matrix(H)

        # Compute received signal
        y_ = received_signal(SNR, x_[s], H)

        # Perform MR receive combining
        xhat_soft_mr = mrc_detection(H, y_)

        # Evaluate MR performance
        ber_mr[case, s] = ber_evaluation(xhat_soft_mr, tx_symbs[s])

        # Perform RZF receive combining
        xhat_soft_rzf, xhat_rzf, Dinv_rzf = rzf_detection(SNR, H, G, y_)

        # Evaluate RZF performance
        ber_rzf[case, s] = ber_evaluation(xhat_soft_rzf, tx_symbs[s])

        # Perform RK-based RZF schemes
        xhat_soft_nrk, xhat_soft_rk, xhat_soft_grk , xhat_soft_rsk = kaczmarz_detection(SNR, H, G, y_, Dinv_rzf, niter_range)

        # Go through each iteration point
        for niter in range(len(niter_range)):
            ber_nrk[case, niter, s] = ber_evaluation(xhat_soft_nrk[niter], tx_symbs[s])
            ber_rk[case, niter, s]  = ber_evaluation(xhat_soft_rk[niter], tx_symbs[s])
            ber_grk[case, niter, s] = ber_evaluation(xhat_soft_grk[niter], tx_symbs[s])
            ber_rsk[case, niter, s] = ber_evaluation(xhat_soft_rsk[niter], tx_symbs[s])

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('mmimo_ber_vs_niter_K'+str(K)+'.npz',
    M=M,
    K=K,
    niter_range=niter_range,
    ber_mr=ber_mr,
    ber_rzf=ber_rzf,
    ber_nrk=ber_nrk,
    ber_rk=ber_rk,
    ber_grk=ber_grk,
    ber_rsk=ber_rsk)

# Compute average values
ber_mr_avg = (ber_mr.mean(axis=-1)).mean(axis=-1)
ber_rzf_avg = (ber_rzf.mean(axis=-1)).mean(axis=-1)
ber_nrk_avg = (ber_nrk.mean(axis=-1)).mean(axis=-1)
ber_rk_avg  = (ber_rk.mean(axis=-1)).mean(axis=-1)
ber_grk_avg = (ber_grk.mean(axis=-1)).mean(axis=-1)
ber_rsk_avg = (ber_rsk.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots(ncols=2)

ax[0].plot(niter_range, ber_mr_avg[0]*np.ones((len(niter_range))), label='MR')
ax[0].plot(niter_range, ber_rzf_avg[0]*np.ones((len(niter_range))), label='RZF')
ax[0].plot(niter_range, ber_nrk_avg[0], label='nRK-RZF')
ax[0].plot(niter_range, ber_rk_avg[0], label='RK-RZF')
ax[0].plot(niter_range, ber_grk_avg[0], label='rGRK-RZF')
ax[0].plot(niter_range, ber_rsk_avg[0], label='RSK-RZF')

ax[0].legend()

ax[0].set_xlabel('iterations')
ax[0].set_ylabel('average BER')

ax[0].set_yscale('log')
ax[0].set_xscale('log', base=2)

ax[1].plot(niter_range, ber_rzf_avg[1]*np.ones((len(niter_range))))
ax[1].plot(niter_range, ber_nrk_avg[1])
ax[1].plot(niter_range, ber_rk_avg[1])
ax[1].plot(niter_range, ber_grk_avg[1])
ax[1].plot(niter_range, ber_rsk_avg[1])

ax[1].set_xlabel('iterations')
ax[1].set_ylabel('average BER')

ax[1].set_yscale('log')
ax[1].set_xscale('log', base=2)

plt.show()
