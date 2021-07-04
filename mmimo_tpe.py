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

########################################
# Preamble
########################################

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
SNRdB_range = np.arange(-10, 11)
SNR_range = 10**(SNRdB_range/10)

########################################
# Simulation parameters
########################################

# Define number of simulation setups
nsetups = 100

# Define number of channel realizations
nchnlreal = 100

# Define maxiter
maxiter = 12

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

    # Generate communication setup
    Huncorr, Hcorr = massive_mimo(M, K, nchnlreal, iota=.5)

    # Go through all different SNR values
    for ss, SNR in enumerate(SNR_range):

        print(f"\tsnr: {ss}/{len(SNR_range)-1}")

        # Go through all channel cases
        for case in range(2):

            if case == 0:
                H = Huncorr
            else:
                H = Hcorr

            # Compute the Gramian matrix
            G = channel_gramian_matrix(H)

            # Compute received signal
            y_ = received_signal(SNR, x_[s], H)

            # Perform RZF receiver
            xhat_soft_rzf, xhat_rzf, Dinv_rzf = rzf_detection(SNR, H, G, y_)

            # Evaluate RZF performance
            ber_rzf[case, ss, s] = ber_evaluation(xhat_soft_rzf, tx_symbs[s])

            # Perform TPE receiver
            xhat_soft_tpe = tpe_detection(SNR, H, G, y_, Dinv_rzf, maxiter=maxiter)

            # Evaluate TPE performance
            ber_tpe[case, ss, s] = ber_evaluation(xhat_soft_tpe, tx_symbs[s])

    print('[setup] elapsed '+str(time.time()-timer_setup)+' seconds.\n')

now = datetime.now()
print(now.strftime("%B %d, %Y -- %H:%M:%S"))
print('--------------------------------------------------')

np.savez('tpe_mmimo_ber_vs_snr_K'+str(K)+'.npz',
    M=M,
    K=K,
    SNRdB_range=SNRdB_range,
    ber_rzf=ber_rzf,
    ber_tpe=ber_tpe)

# Compute average values
ber_rzf_avg = (ber_rzf.mean(axis=-1)).mean(axis=-1)
ber_tpe_avg = (ber_tpe.mean(axis=-1)).mean(axis=-1)

########################################
# Plotting
########################################
fig, ax = plt.subplots()

ax.plot(SNRdB_range, ber_rzf_avg[1], label='RZF')
ax.plot(SNRdB_range, ber_tpe_avg[1], label='nRK-RZF')

ax.legend()

ax.set_xlabel('SNR [dB]')
ax.set_ylabel('average BER')

ax.set_yscale('log')

plt.show()
