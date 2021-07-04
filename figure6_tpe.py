########################################
#   figure6.py
#
#   Description. Script used to generate the curves related to TPE receiver in
#   Figure 6 of the paper.
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

from tpe import *

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

# Range of the number of effective antennas
Drange = np.array([8, 16])

########################################
# Environment parameters
########################################

# Define pre-processing SNR
SNRdB_range = np.arange(-10, 11)
SNR_range = 10**(SNRdB_range/10)

########################################
# Simulation parameters
########################################

# Define number of simulation setups
nsetups = 50

# Define number of channel realizations
nchnlreal = 100

# Obtain maxiter vector
maxiter = 64

########################################
# Running simulation
########################################

# Simulation header
print('--------------------------------------------------')
now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('M-MIMO: BER vs SNR')
print('\t M = '+str(M))
print('\t K = '+str(K))
print('--------------------------------------------------')

# Prepare to save simulation results
ber_rzf = np.zeros((2, SNR_range.size, nsetups, nchnlreal), dtype=np.double)
ber_tpe = np.zeros((2, SNR_range.size, nsetups, nchnlreal), dtype=np.double)

# Obtain qam transmitted signals
tx_symbs, x_ = qam_transmitted_signals(K, nsetups)

# Go through all setups
for s in range(nsetups):

    print(f"setup: {s}/{nsetups-1}")

    timer_setup = time.time()

    # Go through all channel cases
    for case in range(2):

        timer_case = time.time()

        print(f"\tcase: {case}/{2-1}")

        # Generate communication setup
        H = extra_large_mimo(M, K, Drange[case], nchnlreal)

        # Go through all different SNR values
        for ss, SNR in enumerate(SNR_range):

            # Compute the Gramian matrix
            G = channel_gramian_matrix(H)

            # Compute received signal
            y_ = received_signal(SNR, x_[s], H)

            # Perform RZF receive combining
            xhat_soft_rzf, xhat_rzf, Dinv_rzf = rzf_detection(SNR, H, G, y_)

            # Evaluate RZF performance
            ber_rzf[case, ss, s] = ber_evaluation(xhat_soft_rzf, tx_symbs[s])

            # Perform TPE receiver
            xhat_soft_tpe = tpe_detection(SNR, H, G, y_, Dinv_rzf, maxiter=maxiter)

            # Evaluate TPE performance
            ber_tpe[case, ss, s] = ber_evaluation(xhat_soft_tpe, tx_symbs[s])

        print('\t[case] elapsed '+str(time.time()-timer_case)+' seconds.\n')

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('tpe_xlmimo_ber_vs_snr_K'+str(K)+'_'+str(maxiter)+'.npz',
    M=M,
    K=K,
    Drange=Drange,
    SNRdB_range=SNRdB_range,
    maxiter=maxiter,
    ber_rzf=ber_rzf,
    ber_tpe=ber_tpe)

# Compute average values
ber_rzf_avg = (ber_rzf.mean(axis=-1)).mean(axis=-1)
ber_tpe_avg = (ber_tpe.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots()

ax.plot(SNRdB_range, ber_rzf_avg[0], label='RZF')
ax.plot(SNRdB_range, ber_tpe_avg[0], label='nRK-RZF')

ax.legend()

ax.set_xlabel('SNR [dB]')
ax.set_ylabel('average BER')

ax.set_yscale('log')

plt.show()
