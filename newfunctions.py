########################################
#   newfunctions.py
#
#   Description. Set of functions used to evaluate the receivers and implement
#   canonical receivers as Maximum-Ratio (MR), Regularized Zero-Forcing (RZF)
#   schemes.
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
import functools
from joblib import delayed

from kaczmarz import *

########################################
# Private Functions
########################################
def dec2bitarray(in_number, bit_width):
    """
    Convert a positive integer or an array-like of positive integers to NumPy
    array of the specified size containing bits (0 and 1).

    Parameters
    ----------
    in_number : int or array-like of int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """

    if isinstance(in_number, (np.integer, int)):
        return decimal2bitarray(in_number, bit_width).copy()
    result = np.zeros(bit_width * len(in_number), np.int8)
    for pox, number in enumerate(in_number):
        result[pox * bit_width:(pox + 1) * bit_width] = decimal2bitarray(number, bit_width).copy()
    return result

@functools.lru_cache(maxsize=128, typed=False)
def decimal2bitarray(number, bit_width):
    """
    Converts a positive integer to NumPy array of the specified size containing bits (0 and 1). This version is slightly
    quicker that dec2bitarray but only work for one integer.

    Parameters
    ----------
    in_number : int
        Positive integer to be converted to a bit array.
    bit_width : int
        Size of the output bit array.

    Returns
    -------
    bitarray : 1D ndarray of numpy.int8
        Array containing the binary representation of all the input decimal(s).
    """
    result = np.zeros(bit_width, np.int8)
    i = 1
    pox = 0
    while i <= number:
        if i & number:
            result[bit_width - pox - 1] = 1
        i <<= 1
        pox += 1
    return result

########################################
# Public Functions
########################################
def qam_transmitted_signals(K, nsetups):
    """ Generate and modulate user transmitted signals by using 16-QAM.

    Parameters
    ----------
    K : int
        Number of users.

    nsetups : int
        Number of different communication setups.

    Returns
    -------
    tx_symbs : 2D ndarray of np.uint
        Integer generated symbols.
        shape: (nsetups,K)

    tx_basedband_symbs : 2D ndarray of np.cdouble
        Complex-modulated symbols according to constellation.
        shape: (nsetups,K)
    """
    # Define 16-qam constellation vector
    constellation = np.array([-3+1j*3, -3+1j*1, -3-1j*1, -3-1j*3,
                              -1+1j*3, -1+1j*1, -1-1j*1, -1-1j*3,
                              +1+1j*3, +1+1j*1, +1-1j*1, +1-1j*3,
                              +3+1j*3, +3+1j*1, +3-1j*1, +3-1j*3], dtype=np.cdouble)

    # Normalize constellation with respect to average constellation power
    constellation *= np.sqrt(1/10)

    # Modulation order
    m = constellation.shape[0]

    # Generate random transmitted symbols for each user
    tx_symbs = np.random.randint(low=0, high=m, size=(nsetups, K))

    # Perform m-qam modulation
    mapfunc = np.vectorize(lambda i: constellation[i])
    tx_baseband_symbs = (mapfunc(tx_symbs.flatten())).reshape(tx_symbs.shape)

    return tx_symbs, tx_baseband_symbs

def channel_estimates(H, gamma=0.1):
    """ Generate a collection of size nchnlreal of M x K estimated channel
    matrices with a estimation quality of gamma.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of true channel matrices.
        shape: (nchnlreal,M,K)

    gamma : float
        Estimation quality.

    Returns
    -------
    Hhat : 3D ndarray of numpy.cdouble
        Collection of estimated channel matrices.
        shape: (nchnlreal,M,K)
    """
    nchnlreal, M, K = H.shape
    N = np.sqrt(.5)*(np.random.randn(nchnlreal, M, K) + 1j*np.random.randn(nchnlreal, M, K))
    Hhat = np.sqrt(1-(gamma*gamma))*H + gamma*N

    return Hhat

def channel_gramian_matrix(H):
    """ Calculate collection of channel Gramian matrices.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K).

    Returns
    -------
    G : 3D ndarray of numpy.cdouble
        Collection of channel Gramian matrices.
        shape: (nchnlreal,K,K).
    """
    G = np.matmul(H.conj().transpose(0, 2, 1), H)

    return G

def received_signal(SNR, x_, H):
    """Generate base station received signal for each channel realization and
    SNR point.

    Parameters
    ----------
    SNR : float or 1D ndarray of np.double
        Signal-to-noise-ratio values in power units.
        shape: (,) or (len(SNR),)

    x_ : 1D ndarray of numpy.cdouble
        Baseband signals.
        shape: (K,)

    H : 3D ndarray of numpy.cdouble
        Collection of nchnlreal channel matrices.
        shape: (nchnlreal,M,K)

    Returns
    -------
    ySNR : 2D or 3D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M) or (lenght(SNR),nchnlreal,M)

    Notes
    -----
    Function considers white complex Gaussian noise.
    """
    nchnlreal, M, K = H.shape

    if isinstance(SNR, float):
        SNR = np.array([SNR])

    # Generate white complex-Gaussian noise
    wcgn = np.sqrt(.5)*(np.random.randn(nchnlreal, M) + 1j*np.random.randn(nchnlreal, M))

    # Received signals w/o noise
    no_noise_rx_signal = (H*x_[None, None, :]).sum(axis=-1)

    # SNR's reciprocals
    rec_SNR = np.reciprocal(np.sqrt(SNR))

    # Received signal vector
    ySNR = no_noise_rx_signal[None, :, :] + rec_SNR[:, None, None]*wcgn

    # Get rid of additional dimensions when SNR is a float
    ySNR = np.squeeze(ySNR)

    return ySNR

def mrc_detection(H, y_):
    """ Obtain maximum-ratio combining (MRC) soft signal estimates.

    Parameters
    ----------
    H : 3D ndarray of numpy.cdouble
        Collection of nchnlreal channel matrices.
        shape: (nchnlreal,M,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection of nchnlreal 1D received signals of length M.
        shape: (nchnlreal,M)

    Returns
    -------
    xhat_soft : 1D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)
    """
    Vbar = H / (np.linalg.norm(H, axis=1)**2)[:, None, :]
    xhat_soft = np.squeeze(np.matmul(Vbar.conj().transpose(0, 2, 1), y_[:, :, None]))

    return xhat_soft

def rzf_detection(SNR, H, G, y_):
    """ Obtain regularized zero-forcing (RZF) soft signal estimates. Raw signal
    estimates are outputted for comparison with methods that emulate the RZF
    scheme. Soft normalization matrix Dinv is also outputted.

    Parameters
    ----------
    SNR : float
        Signal-to-noise-ratio in power units.

    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    G : 3D ndarray of numpy.cdouble
        Collection of channel Gramian matrices.
        shape: (nchnlreal,K,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection o received signals.
        shape: (nchnlreal,M)

    Returns
    -------
    xhat_soft : 1D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)

    xhat : 1D ndarray of numpy.cdouble
        Raw signal estimates.
        shape: (nchnlreal,K)

    Dinv : 2D ndarray of numpy.cdouble
        Soft power normalization.
        shape: (nchnlreal,K)
    """
    from numpy.dual import inv

    nchnlreal, M, K = H.shape

    # Constants
    xi = 1/SNR
    eyeK = np.eye(K)

    # Store inverted covariance of the received signal
    Ryy_inv = inv(G + (xi*eyeK)[None, :, :])

    # Compute receive combining matrices
    V = np.matmul(H, Ryy_inv)

    # Store norm of RZF receive combining
    D = np.diagonal(np.matmul(Ryy_inv, G), axis1=1, axis2=2)
    Dinv = np.reciprocal(D)

    # Get RZF signal estimates
    xhat = np.squeeze(np.matmul(V.conj().transpose(0, 2, 1), y_[:, :, None]))

    # Get soft RZF signal estimates
    xhat_soft = Dinv*xhat

    return xhat_soft, xhat, Dinv

def kaczmarz_detection(SNR, H, G, y_, Dinv, niter_range, parl=None):
    """ Obtain user signal estimates by relying on RK-RZF and GRK_RZF.

    Parameters
    ----------
    SNR : float
        Signal-to-noise-ratio in power units

    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    G : 3D ndarray of numpy.cdouble
        Collection of channel Gramian matrices.
        shape: (nchnlreal,K,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M)

    Dinv : 2D ndarray of numpy.cdouble
        Soft power normalization.
        shape: (nchnlreal,K)

    parl : Parallel object from joblib
        If not none, parallel computing is realized. parl object is imported
        from joblib module.

    Returns
    -------
    xhat_soft_nrk : 3D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (len(rtol), nchnlreal, K) or (nchnlreal, K)

    xhat_soft_rk : 3D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (len(rtol), nchnlreal, K) or (nchnlreal, K)

    xhat_soft_grk : 3D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (len(rtol), nchnlreal, K) or (nchnlreal, K)

    xhat_soft_rsk : 3D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (len(rtol), nchnlreal, K) or (nchnlreal, K)

    """
    nchnlreal, M, K = H.shape

    # Constants
    xi = 1/SNR

    # Prepare outputs
    len_niter = len(niter_range)

    xhat_nrk = np.zeros((len_niter, nchnlreal, K), dtype=np.cdouble)
    xhat_rk  = np.zeros((len_niter, nchnlreal, K), dtype=np.cdouble)
    xhat_grk = np.zeros((len_niter, nchnlreal, K), dtype=np.cdouble)
    xhat_rsk = np.zeros((len_niter, nchnlreal, K), dtype=np.cdouble)

    if parl is not None:
        results_nrk = parl(delayed(nrk_rzf_iteration)(H[n], G[n], y_[n], xi, niter_range=niter_range) for n in range(nchnlreal))
        results_rk  = parl(delayed(rk_rzf_iteration)(H[n], G[n], y_[n], xi, niter_range=niter_range) for n in range(nchnlreal))
        results_grk = parl(delayed(grk_rzf_iteration)(H[n], G[n], y_[n], xi, niter_range=niter_range) for n in range(nchnlreal))
        results_rsk = parl(delayed(rsk_rzf_iteration)(H[n], G[n], y_[n], xi, niter_range=niter_range) for n in range(nchnlreal))

        # Extract parallel results
        xhat_nrk = np.array(results_nrk).transpose(1, 0, 2)
        xhat_rk  = np.array(results_rk).transpose(1, 0, 2)
        xhat_grk = np.array(results_grk).transpose(1, 0, 2)
        xhat_rsk = np.array(results_rsk).transpose(1, 0, 2)

    else:
        for n in range(nchnlreal):
            xhat_nrk[:, n] = nrk_rzf_iteration(H[n], G[n], y_[n], xi, niter_range)
            xhat_rk[:, n]  = rk_rzf_iteration(H[n], G[n], y_[n], xi, niter_range)
            xhat_grk[:, n] = grk_rzf_iteration(H[n], G[n], y_[n], xi, niter_range)
            xhat_rsk[:, n] = rsk_rzf_iteration(H[n], G[n], y_[n], xi, niter_range)

    # Get soft estimates
    xhat_soft_nrk = Dinv[None, :, :]*xhat_nrk
    xhat_soft_rk  = Dinv[None, :, :]*xhat_rk
    xhat_soft_grk = Dinv[None, :, :]*xhat_grk
    xhat_soft_rsk = Dinv[None, :, :]*xhat_rsk

    return xhat_soft_nrk, xhat_soft_rk, xhat_soft_grk, xhat_soft_rsk

def kaczmarz_detection_maxiter(SNR, H, G, y_, Dinv, maxiter, parl=None):
    """ Obtain user signal estimates by relying on RK-RZF and GRK_RZF.

    Parameters
    ----------
    SNR : float
        Signal-to-noise-ratio in power units

    H : 3D ndarray of numpy.cdouble
        Collection of channel matrices.
        shape: (nchnlreal,M,K)

    G : 3D ndarray of numpy.cdouble
        Collection of channel Gramian matrices.
        shape: (nchnlreal,K,K)

    y_ : 2D ndarray of numpy.cdouble
        Collection of received signals.
        shape: (nchnlreal,M)

    Dinv : 2D ndarray of numpy.cdouble
        Soft power normalization.
        shape: (nchnlreal,K)

    parl : Parallel object from joblib
        If not none, parallel computing is realized. parl object is imported
        from joblib module.

    Returns
    -------
    xhat_soft_nrk : 2D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal, K)

    xhat_soft_rk : 2D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal, K)

    xhat_soft_grk : 2D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal, K)

    xhat_soft_rsk : 2D ndarrays of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal, K)

    """
    nchnlreal, M, K = H.shape

    # Constants
    xi = 1/SNR

    xhat_nrk = np.zeros((nchnlreal, K), dtype=np.cdouble)
    xhat_rk  = np.zeros((nchnlreal, K), dtype=np.cdouble)
    xhat_grk = np.zeros((nchnlreal, K), dtype=np.cdouble)
    xhat_rsk = np.zeros((nchnlreal, K), dtype=np.cdouble)

    if parl is not None:
        results_nrk = parl(delayed(nrk_rzf_iteration)(H[n], G[n], y_[n], xi, maxiter=maxiter[0]) for n in range(nchnlreal))
        results_rk  = parl(delayed(rk_rzf_iteration)(H[n], G[n], y_[n], xi, maxiter=maxiter[1]) for n in range(nchnlreal))
        results_grk = parl(delayed(grk_rzf_iteration)(H[n], G[n], y_[n], xi,  maxiter=maxiter[2]) for n in range(nchnlreal))
        results_rsk = parl(delayed(rsk_rzf_iteration)(H[n], G[n], y_[n], xi,  maxiter=maxiter[3]) for n in range(nchnlreal))

        # Extract parallel results
        xhat_nrk = np.array(results_nrk)
        xhat_rk  = np.array(results_rk)
        xhat_grk = np.array(results_grk)
        xhat_rsk = np.array(results_rsk)

    else:
        for n in range(nchnlreal):
            xhat_nrk[n] = nrk_rzf_iteration(H[n], G[n], y_[n], xi, maxiter=maxiter[0])
            xhat_rk[n] = rk_rzf_iteration(H[n], G[n], y_[n], xi, maxiter=maxiter[1])
            xhat_grk[n] = grk_rzf_iteration(H[n], G[n], y_[n], xi, maxiter=maxiter[2])
            xhat_rsk[n] = rsk_rzf_iteration(H[n], G[n], y_[n], xi, maxiter=maxiter[3])

    # Get soft estimates (get rid of channel norms)
    xhat_soft_nrk = Dinv*xhat_nrk
    xhat_soft_rk  = Dinv*xhat_rk
    xhat_soft_grk = Dinv*xhat_grk
    xhat_soft_rsk = Dinv*xhat_rsk

    return xhat_soft_nrk, xhat_soft_rk, xhat_soft_grk, xhat_soft_rsk

def qam_received_signals(xsoft):
    """ Perform m-QAM demodulation based on the hard threshold detector.
    Return nearest demodulated symbols.

    Parameters
    ----------
    xsoft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)

    Returns
    -------
    rx_symbs : 2D ndarray of np.uint
        Demodulated symbols.
        shape: (nchnlreal,K)
    """
    # Define 16-qam constellation vector
    constellation = np.array([-3+1j*3, -3+1j*1, -3-1j*1, -3-1j*3,
                              -1+1j*3, -1+1j*1, -1-1j*1, -1-1j*3,
                              +1+1j*3, +1+1j*1, +1-1j*1, +1-1j*3,
                              +3+1j*3, +3+1j*1, +3-1j*1, +3-1j*3], dtype=np.cdouble)

    # Normalize constellation with respect to average constellation power
    constellation *= np.sqrt(1/10)

    # Perform hard-threshold demodulation
    rx_symbs = np.abs(xsoft[:, :, None] - constellation).argmin(-1)

    return rx_symbs

def ber_evaluation(xsoft, tx_symbs):
    """ Count the bit error rate (BER) per user.

    Parameters
    ----------
    xsoft : 2D ndarray of numpy.cdouble
        Soft signal estimates.
        shape: (nchnlreal,K)

    tx_symbs : 1D ndarray of np.uint
        True integer generated symbols.
        shape: (K,)

    Returns
    -------
    ber_peruser : 1D ndarray of np.cdouble
        BER per user.
        shape: (nchnlreal,)
    """
    nchnlreal, K = xsoft.shape
    num_bits_symb = 4

    rx_symbs = qam_received_signals(xsoft)

    tx_bits = dec2bitarray(tx_symbs, num_bits_symb)
    rx_bits = dec2bitarray(rx_symbs.ravel(), num_bits_symb)

    hamming_distance = np.bitwise_xor(np.tile(tx_bits, nchnlreal), rx_bits).reshape(nchnlreal, num_bits_symb*K).sum(axis=-1)

    ber_peruser = hamming_distance / (num_bits_symb*K)

    return ber_peruser
