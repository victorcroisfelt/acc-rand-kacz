########################################
#   commsetup.py
#
#   Description. Set of functions to generate communication setups: Massive MIMO
#   (M-MIMO) and Extra-Large Scale MIMO (XL-MIMO).
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

########################################
# Private functions
########################################
def nonstationary(M, K, D):
    """ Output the diagonal matrix containing non-stationary effects.

    Parameters
    ----------
    M : int
        Number of antennas.

    K : int
        Number of users.

    D : int
        Number of visible antennas per user.

    Returns
    -------
    diag : 3D ndarray of numpy.cdouble
        Collection of diagonal matrices.
        shape: (K,M,M)
    """

    # Non-stationary
    diag = np.zeros((M, K), dtype=np.cdouble)

    if D != M:  # XL-MIMO

        # Generating VRs
        offset = D//2
        centers = np.random.randint(M, size=K, dtype=np.int)

        if D%2 == 0:
            upper = centers + offset
        else:
            upper = centers + (offset + 1)

        lower = centers - offset

        # Check correctness of the sets
        upper[upper >= M] = M-1
        lower[lower < 0] = 0

        # Generating diagonal matrices
        zerosM = np.zeros(M, dtype=np.cdouble)
        for k in range(K):
            els = zerosM.copy()
            els[lower[k]:upper[k]] = 1

            diag[:, k] = els

        # Normalization
        diag *= M/D

    else:   # M-MIMO
        diag = np.ones((M, K))

    return diag

########################################
# Public Functions
########################################
def massive_mimo(M, K, nchnlreal, iota=0.5):
    """ Generate an M-MIMO communication setup.

    Parameters
    ----------
    M : int
        Number of antennas.

    K : int
        Number of users.

    nchnlreal : int
        Number of channel realizations.

    iota : float
        Level of spatial correlation.

    Returns
    -------
    Huncorr : 3D ndarray of numpy.cdouble
        Collection of uncorrelated channel matrices.
        shape: (nchnlreal,M,K)

    Hcorr : 3D ndarray of numpy.cdouble
        Collection of correlated channel matrices.
        shape: (nchnlreal,M,K)
    """
    from scipy.linalg import toeplitz
    from numpy.dual import svd

    # Define square length
    squareLength = 400

    # Define BS position
    BSposition = squareLength/2 + 1j*squareLength/2

    # Define minimum distance
    min_distance = 35

    # Distribute the users in cell area
    user_positions = (np.random.rand(K) + 1j*np.random.rand(K))*squareLength

    # Check distance
    while True:

        # Distance of the users to the BS
        distances = abs(BSposition - user_positions)

        # Get indexes from users too near to BS
        user_index = np.where(distances < min_distance)

        # Get the length of the user index
        nusers = len(user_index[0])

        # Check if everything is ok
        if nusers == 0:
            break

        # Redistribution
        user_positions[user_index] = (np.random.rand(nusers) + 1j*np.random.rand(nusers))*squareLength

    # Compute final distances of the users
    distances = abs(BSposition - user_positions)

    # Compute channel gains
    channel_gains = 10**((10 + 96 - 30.5 - 36.7 * np.log10(distances))/10)

    # Prepare to save correlated channel matrices
    Rcorr = np.empty((K, M, M), np.cdouble)

    # Generate correlated channel matrices
    for k in range(K):

        # Compute 1st col
        col = iota**np.arange(M)

        # Compute matrix
        Rcorr[k] = toeplitz(c=col)

    # Generate uncorrelated channels
    Huncorr = np.sqrt(channel_gains[None, None, :]/2)*(np.random.randn(nchnlreal, M, K) + 1j*np.random.randn(nchnlreal, M, K))

    # Factorizing covariance matrices
    (_, s_, vh_) = svd(Rcorr)

    c_ = np.sqrt(s_[:, :, None]) * vh_
    c_ = c_.conj().transpose(0, 2, 1)

    # Generate correlated channels
    Hcorr = np.matmul(c_, Huncorr.T).T

    return Huncorr, Hcorr


def extra_large_mimo(M, K, D, nchnlreal):
    """ Generate a XL-MIMO communication setup.

    Parameters
    ----------
    M : int
        Number of antennas.

    K : int
        Number of users.

    D : int
        Number of visible antennas.

    nchnlreal : int
        Number of channel realizations.

    iota : float
        Level of spatial correlation.

    Returns
    -------
    Huncorr : 3D ndarray of numpy.cdouble
        Collection of uncorrelated channel matrices.
        shape: (nchnlreal,M,K)
    """

    # Define the length of the antenna array
    L = 250

    # Obtain the distance between the elements of the antenna array
    delta = L/M

    # Obtain a vector with the position of each antenna element
    antennaPositions = np.linspace(0, L, M) + 1j*0

    # Define guard zone of y
    guardZone = 0.1*L

    # Distribute the users in the front of the antenna array
    user_positions = (np.random.rand(K) + 1j*np.random.rand(K))*L

    # Check guard distance
    while True:

        # Get indexes from users very close to the antenna array
        user_index = np.where(user_positions.imag < guardZone)

        # Get the length of the user index
        nusers = len(user_index[0])

        # Check if everything is ok
        if nusers == 0:
            break

        # Redistribution
        user_positions[user_index] = (np.random.rand(nusers) + 1j*np.random.rand(nusers))*L

    # Compute an M x K matrix with the distance of each user towards all the antenna elements
    distances = abs(antennaPositions[:, None] - user_positions)

    # Compute an M x K matrix containing the channel gains
    channel_gains =  10**((10 + 96 - 30.5 - 36.7 * np.log10(distances))/10)

    # Obtain the diagonal matrix containing non-stationary information
    Dmatrix = nonstationary(M, K, D)

    # Generate uncorrelated channels
    Huncorr = np.sqrt((Dmatrix*channel_gains/2))*(np.random.randn(nchnlreal, M, K) + 1j*np.random.randn(nchnlreal, M, K))

    return Huncorr
