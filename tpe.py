########################################
#   tpe.py
#
#   Description. Set of functions related to the implementation of TPE receiver.
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
import math

########################################
# Private functions
########################################
def nCk(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

########################################
# Public functions
########################################
def tpe_detection_optz(SNR, H, G, y_, Dinv, Jrange):
    """ Implementation of the optimal TPE receiver.

    Parameters
    ----------
    SNR : float
        Signal-to-noise ratio.

    H : 3D ndarray of numpy.cdouble
        Collection of channel matrix.
        shape: (nchnlreal,M,K)

    G : 3D ndarray of numpy.cdouble
        Collection of Gramian matrix.
        shape: (nchnlreal,K,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M)

    Dinv : 3D ndarray of numpy.cdouble
        Collection of re-scaling matrices.
        shape: (nchnlreal,K,K)

    Jrange : 1D ndarray of integers
        Parameter J of TPE.

    Returns
    -------

    xhat_soft : 2D ndarray of numpy.cdouble
        Collection of soft estimates.
        shape: (nchnlreal, K)
     """

    # Extract dimensionns
    nchnlreal, M, K = H.shape

    # Compute inverse of the SNR
    xi = 1/SNR

    # Extract eigenvalues from the Gramian matrix
    W_, _ = np.linalg.eig(G)

    # Compute alpha_opt
    alpha_opt = 2/(W_[:, 0] + W_[:, -1] + 2*xi)

    # Compute MR estimates
    yMR = np.squeeze(np.matmul(H.conj().transpose(0, 2, 1), y_[:, :, None]))

    # Prepare to obtain the estimated symbols
    xhat = alpha_opt[:, None] * yMR.copy()

    # Go through all channel realizations
    for real in range(nchnlreal):

        # Go through each iteration
        for j in range(Jrange):

            #Store X matrix
            X_ = np.eye(K) - alpha_opt[real] * (G[real] + xi*np.eye(K))

            # Compute a new approximation
            xhat[real] = alpha_opt[real] * yMR[real] + X_ @ xhat[real]

    # Rescale the power
    xhat_soft = Dinv*xhat

    return xhat_soft

def tpe_detection(SNR, H, G, y_, Dinv, niter_range=None, maxiter=None):
    """ Implementation of the canonical RPE.

    Parameters
    ----------
    SNR : float
        Signal-to-noise ratio.

    H : 3D ndarray of numpy.cdouble
        Collection of channel matrix.
        shape: (nchnlreal,M,K)

    G : 3D ndarray of numpy.cdouble
        Collection of Gramian matrix.
        shape: (nchnlreal,K,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M)

    Dinv : 3D ndarray of numpy.cdouble
        Collection of re-scaling matrices.
        shape: (nchnlreal,K,K)

    niter_range : 1D ndarray of integers
        Number of iterations.

    maxiter : 1D ndarray of integers
        Maximum number of iterations.

    Returns
    -------

    xhat_soft : 2D ndarray of numpy.cdouble
        Collection of soft estimates.
        shape: (nchnlreal, K)
     """

    # Extract dimensionns
    nchnlreal, M, K = H.shape

    # Compute inverse of the SNR
    xi = 1/SNR

    # Check entries
    if maxiter is None:

        # Length of the range of the number of iterations
        len_niter = len(niter_range)

        # Extract maxiter
        maxiter = niter_range[-1]

    # Define an arbitrarily small number
    epsilon = 1e-6

    # Compute the row sum of the Gramian matrix
    rowSums = np.abs(G).sum(axis=2)

    # Estimate the max and min eigenvalues
    lambdaHat_max = rowSums.max(axis=1)
    lambdaHat_min = np.maximum((np.diagonal(G, axis1=1, axis2=2) - rowSums).min(axis=1), epsilon)

    # Estimated alpha
    alpha_est = 2/(lambdaHat_max + lambdaHat_min + 2*xi)

    # Compute MR estimates
    yMR = np.squeeze(np.matmul(H.conj().transpose(0, 2, 1), y_[:, :, None]))

    # Prepare to save soft estimates
    if niter_range is None:
        xhat = np.empty((nchnlreal, K), dtype=np.cdouble)
    else:
        xhat = np.empty((len_niter, nchnlreal, K), dtype=np.cdouble)

    # Go through all channel realizations
    for real in range(nchnlreal):

        # Start number of iterations counter
        t = 0

        # Initialize the estimated symbols
        x_ = alpha_est[real, None] * yMR[real]

        #Store X matrix
        X_ = np.eye(K) - alpha_est[real] * (G[real] + xi*np.eye(K))

        # Iterative process
        while True:

            # Compute a new approximation
            x_ = alpha_est[real] * yMR[real] + X_ @ x_

            # Update iteration counter
            t += 1

            if niter_range is None:

                # Check maxiter
                if t == maxiter:

                    # Save result
                    xhat[real] = x_.copy()

                    break

            else:

                # Check condition
                if np.sum(t == niter_range) > 0:

                    # Get the index from niter_range
                    index = np.where(t == niter_range)[0][0]

                    # Save result
                    xhat[index, real] = x_.copy()

                    # Check maxiter
                    if t == maxiter:
                        break

    if niter_range is None:

        # Rescale the power
        xhat_soft = Dinv*xhat

    else:

        xhat_soft = Dinv[None, :, :]*xhat

    return xhat_soft
