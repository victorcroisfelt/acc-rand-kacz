########################################
#   kaczmarz.py
#
#   Description. Set of functions related to Kaczmarz-based receivers.
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
def rand_discrete_variate(p_):
    """Draw an integer, discrete sample from a user-defined distribution. Only
    returns one sample.

    Parameters
    ----------
    p_ : 1D ndarray of np.double
        Vector with probabilities. Sum of its elements needs to be allclose to
        1.0. Function checks if this is true. If not, it raises an error.

    Returns
    -------
    rand : int
        Randomly generated integer based on p_.

    Raises
    ------
    ValueError
        If sum of p_ is not close to 1.0.

    Notes
    -----
    If there are zero entries in p_, the integers associated with them are
    disregarded. Possible values are defined by using the valid_indices vector.
    """
    if not np.allclose(np.sum(p_), 1.0):
        raise ValueError(
            "the probability vector p_ is not a valid one."
        )

    valid_indices = np.flatnonzero(p_)
    pnew = p_[valid_indices]
    rand_index = sum(np.random.rand() > pnew.cumsum())
    rand = valid_indices[rand_index]

    return rand

########################################
# Public Functions
########################################
def nrk_rzf_iteration(H, G, y_, xi, niter_range=None, maxiter=None):
    """ Estimate soft estimates using nRK-RZF.

    Parameters
    ----------
    H : 2D ndarray of numpy.cdouble
        Channel matrix.
        shape: (M,K)

    G : 2D ndarray of numpy.cdouble
        Channel Gramian matrix.
        shape: (K,K)

    y : 1D ndarray of numpy.cdouble
        Received signal.
        shape: (M,)

    xi : float
        Inverse of SNR.

    maxiter : int
        Maximum number of iterations. *Default" maxiter=None.

    Returns
    -------
    xhat : 2D ndarray of numpy.cdouble
        Soft estimates.
        shape: (len(niter_range),K)

    Notes
    -----
    p_ stands for sampling probability vector.

    Alias
    -----
    sql2norm - squared l2-norm of a vector.
    """
    M, K = H.shape

    # Pre-processing (constants)
    b_ = (H.conj().T*y_[None, :]).sum(-1)
    sql2norm_eq = np.diag(G).real + xi

    # Reciprocals
    rec_eq = np.reciprocal(sql2norm_eq)

    # Check entries
    if maxiter is None:

        # Length of the range of the number of iterations
        len_niter = len(niter_range)

        # Extract maxiter
        maxiter = niter_range[-1]

    # Initialization
    u_ = np.zeros(M, dtype=np.cdouble)
    v_ = np.zeros(K, dtype=np.cdouble)

    # Sampling probability vector
    p_ = sql2norm_eq/sql2norm_eq.sum()

    # Vector with random selected equations
    it_vec = np.random.choice(K, size=maxiter, replace=True, p=p_)

    # Start number of iterations counter
    t = 0

    # Prepare to save soft estimates
    if niter_range is None:
        xhat = np.empty((K), dtype=np.cdouble)
    else:
        xhat = np.empty((len_niter, K), dtype=np.cdouble)

    # Iterative process
    while True:

        # Iterative step
        it = it_vec[t]

        # Kaczmarz step
        r_it = b_[it] - (H[:, it].conj()*u_).sum() - xi*v_[it]
        gamma = rec_eq[it]*r_it

        # Update step
        u_ += gamma*H[:, it]
        v_[it] += gamma

        # Update iteration counter
        t += 1

        if niter_range is None:

            # Check maxiter
            if t == maxiter:

                # Save result
                xhat = v_.copy()

                break

        else:

            # Check condition
            if np.sum(t == niter_range) > 0:

                # Get the index from niter_range
                index = np.where(t == niter_range)[0][0]

                # Save result
                xhat[index] = v_.copy()

                # Check maxiter
                if t == maxiter:
                    break

    return xhat

def rk_rzf_iteration(H, G, y_, xi, niter_range=None, maxiter=None):
    """ Estimate user signal by using RK-RZF.

    Parameters
    ----------
    H : 2D ndarray of numpy.cdouble
        Channel matrix.
        shape: (M,K)

    G : 2D ndarray of numpy.cdouble
        Channel Gramian matrix.
        shape: (K,K)

    y : 1D ndarray of numpy.cdouble
        Received signal.
        shape: (M,)

    xi : float
        Inverse of SNR.

    x_ : 1D ndarray of numpy.cdouble
        RZF user signal estimates.
        shape: (K,)

    rtol : float or 1D ndarray of numpy.cdouble
        Relative tolerance. *Default* rtol=1e-6.

    maxiter : int
        Maximum number of iterations. *Default" maxiter=None.

    Returns
    -------
    xhat : 1D or 2D ndarray of numpy.cdouble
        User signal estimates.
        shape: (K,) or (len(rtol),K)

    niter : int or 1D ndarray of numpy.uint
        Number of iterations to convergence.
        shape: (len(rtol),)

    Notes
    -----
    p_ stands for probability vector.

    Alias
    -----
    sql2norm - squared l2-norm of a vector.
    """
    M, K = H.shape

    # Pre-processing (constants)
    b_ = (H.conj().T*y_[None, :]).sum(-1)
    sql2norm_eq = np.diag(G).real + xi

    # Reciprocals
    rec_eq = np.reciprocal(sql2norm_eq)

    # Check entries
    if maxiter is None:

        # Length of the range of the number of iterations
        len_niter = len(niter_range)

        # Extract maxiter
        maxiter = niter_range[-1]

    # Initialization
    u_ = np.zeros(M, dtype=np.cdouble)
    v_ = np.zeros(K, dtype=np.cdouble)

    # Sampling probability vector
    p_ = sql2norm_eq/sql2norm_eq.sum()

    # Start number of iterations counter
    t = 0

    # Prepare to save soft estimates
    if niter_range is None:
        xhat = np.empty((K), dtype=np.cdouble)
    else:
        xhat = np.empty((len_niter, K), dtype=np.cdouble)

    # Iterative process
    while True:

        # Sweep definition
        if (t%K) == 0:
            it_vect = list(np.random.choice(K, size=K, replace=False, p=p_))

        # Iterative step
        it = it_vect.pop(0)

        # Kaczmarz step
        r_it = b_[it] - (H[:, it].conj()*u_).sum() - xi*v_[it]
        gamma = rec_eq[it]*r_it

        # Update step
        u_ += gamma*H[:, it]
        v_[it] += gamma

        # Update iteration counter
        t += 1

        if niter_range is None:

            # Check maxiter
            if t == maxiter:

                # Save result
                xhat = v_.copy()

                break

        else:

            # Check condition
            if np.sum(t == niter_range) > 0:

                # Get the index from niter_range
                index = np.where(t == niter_range)[0][0]

                # Save result
                xhat[index] = v_.copy()

                # Check maxiter
                if t == maxiter:
                    break

    return xhat

def grk_rzf_iteration(H, G, y_, xi, niter_range=None, maxiter=None):
    """ Estimate user signal by using GRK-RZF.

    Parameters
    ----------
    H : 2D ndarray of numpy.cdouble
        Channel matrix.
        shape: (M,K).

    G : 2D ndarray of numpy.cdouble
        Channel Gramian matrix.
        shape: (K,K).

    y : 1D ndarray of numpy.cdouble
        Received signal.
        shape: (M,).

    xi : float
        Inverse of SNR.

    x_ : 1D ndarray of numpy.cdouble
        RZF user signal estimates.
        shape:(K,)

    rtol : float or 1D ndarray of numpy.cdouble
        Relative tolerance. *Default* rtol=1e-6.

    maxiter : int
        Maximum number of iterations. *Default" maxiter=None.

    Returns
    -------
    xhat : 1D or 2D ndarray of numpy.cdouble
        User signal estimates.
        shape: (K,) or (len(rtol),K)

    niter : int or 1D ndarray of numpy.uint
        Number of iterations to convergence.
        shape: (len(rtol),)

    Notes
    -----
    G stands for the channel Gramian matrix. The diagonal of G plus xi is the
    squared l2-norms of the equations. p_ vector stands for probability vector.

    Alias
    -----
    sql2norm - squared l2-norm of a vector.
    sqfrobnorm - squared Frobenius norm of a matrix.
    RS - residual squares.
    RSS - residual sum of squares.
    """
    M, K = H.shape

    # Pre-processing (constants)
    b_ = (H.conj().T*y_[None, :]).sum(-1)

    Ryy = G + xi*np.eye(K)

    sql2norm_eq = np.diag(Ryy).real
    sqfrobnorm_Bh = sql2norm_eq.sum()

    # Reciprocals
    rec_eq = np.reciprocal(sql2norm_eq)
    rec_Bh = np.reciprocal(sqfrobnorm_Bh)

    # Check entries
    if maxiter is None:

        # Length of the range of the number of iterations
        len_niter = len(niter_range)

        # Extract maxiter
        maxiter = niter_range[-1]

    # Initialization
    u_ = np.zeros(M, dtype=np.cdouble)
    v_ = np.zeros(K, dtype=np.cdouble)
    r_ = b_.copy()

    # Sampling probability vector
    p_ = sql2norm_eq/sql2norm_eq.sum()

    # Start number of iterations counter
    t = 0

    # Prepare to save soft estimates
    if niter_range is None:
        xhat = np.empty((K), dtype=np.cdouble)
    else:
        xhat = np.empty((len_niter, K), dtype=np.cdouble)

    # Iterative process
    while True:

        # Iterative constants
        absr_ = np.abs(r_)
        RS = absr_*absr_
        RSS = RS.sum()

        # Step 01: determine set of working (active) equations
        epsilon = 0.5*(((rec_eq*RS).max())/RSS + rec_Bh)
        constant = epsilon*RSS
        not_working_mask = np.where(RS < constant*sql2norm_eq)

        # Step 02: obtain p_ vector
        RStilde = RS.copy()
        RStilde[not_working_mask] = 0.0

        p_ = RStilde/RStilde.sum()

        # Step 03: Kaczmarz projection
        it = rand_discrete_variate(p_)

        # Kaczrmaz step
        gamma = rec_eq[it]*r_[it]

        # Update step
        u_ += gamma*H[:, it]
        v_[it] += gamma
        r_ -= gamma*Ryy[:, it]

        # Update iteration counter
        t += 1

        if niter_range is None:

            # Check maxiter
            if t == maxiter:

                # Save result
                xhat = v_.copy()

                break

        else:

            # Check condition
            if np.sum(t == niter_range) > 0:

                # Get the index from niter_range
                index = np.where(t == niter_range)[0][0]

                # Save result
                xhat[index] = v_.copy()

                # Check maxiter
                if t == maxiter:
                    break

    return xhat

def rsk_rzf_iteration(H, G, y_, xi, niter_range=None, maxiter=None):
    """ Estimate user signal by using RSK-RZF.

    Parameters
    ----------
    H : 2D ndarray of numpy.cdouble
        Channel matrix.
        shape: (M,K)

    G : 2D ndarray of numpy.cdouble
        Channel Gramian matrix.
        shape: (K,K)

    y : 1D ndarray of numpy.cdouble
        Received signal.
        shape: (M,)

    xi : float
        Inverse of SNR.

    x_ : 1D ndarray of numpy.cdouble
        RZF user signal estimates.
        shape: (K,)

    rtol : float or 1D ndarray of numpy.cdouble
        Relative tolerance. *Default* rtol=1e-6.

    maxiter : int
        Maximum number of iterations. *Default" maxiter=None.

    omega : int
        Size of the set of working equations.

    Returns
    -------
    xhat : 1D or 2D ndarray of numpy.cdouble
        User signal estimates.
        shape: (K,) or (len(rtol),K)

    niter : int or 1D ndarray of numpy.uint
        Number of iterations to convergence.
        shape: (len(rtol),)

    Notes
    -----
    p_ stands for probability vector.

    Alias
    -----
    sql2norm - squared l2-norm of a vector.
    """
    M, K = H.shape

    omega = None

    if omega is None:
        omega = np.ceil(np.log2(K)).astype(np.int)

    # Pre-processing (constants)
    b_ = (H.conj().T*y_[None, :]).sum(-1)

    sql2norm_eq = np.diag(G).real + xi
    sqfrobnorm_Bh = sql2norm_eq.sum()

    # Reciprocals
    rec_eq = np.reciprocal(sql2norm_eq)
    rec_Bh = np.reciprocal(sqfrobnorm_Bh)

    # Check entries
    if maxiter is None:

        # Length of the range of the number of iterations
        len_niter = len(niter_range)

        # Extract maxiter
        maxiter = niter_range[-1]

    # Initialization
    u_ = np.zeros(M, dtype=np.cdouble)
    v_ = np.zeros(K, dtype=np.cdouble)

    # Sampling probability vector
    p_ = sql2norm_eq/sql2norm_eq.sum()

    # Start number of iterations counter
    t = 0

    # Prepare to save soft estimates
    if niter_range is None:
        xhat = np.empty((K), dtype=np.cdouble)
    else:
        xhat = np.empty((len_niter, K), dtype=np.cdouble)

    # Iterative process
    while True:

        # Getting the set of working equations
        working_eqs = np.random.choice(K, size=(omega), replace=False)

        # Compute residuals of working equations
        r_eqs = np.zeros(K, dtype=np.cdouble)

        for w in working_eqs:
            r_eqs[w] = b_[w] - (H[:, w].conj()*u_).sum() - xi*v_[w]

        # Taking the absolute values and squaring
        absr_eqs = np.abs(r_eqs)
        RS_eqs = absr_eqs*absr_eqs
        RR_eqs = rec_Bh*RS_eqs

        # Find the index of the maximum residual
        it = RR_eqs.argmax()

        # Kaczmarz step
        gamma = rec_eq[it]*r_eqs[it]

        # Update
        u_ += gamma*H[:, it]
        v_[it] += gamma

        # Update iteration counter
        t += 1

        if niter_range is None:

            # Check maxiter
            if t == maxiter:

                # Save result
                xhat = v_.copy()

                break

        else:

            # Check condition
            if np.sum(t == niter_range) > 0:

                # Get the index from niter_range
                index = np.where(t == niter_range)[0][0]

                # Save result
                xhat[index] = v_.copy()

                # Check maxiter
                if t == maxiter:
                    break

    return xhat

def kaczmarz_maxiter(M, K, omega=None):
    """ Compute maximum number of iterations for
        1.nRK-RZF
        2.RK-RZF
        3.GRK-RZF
        4.RSK-RZF

    Parameters
    ----------
    M : int
        Number of antennas.

    K : int
        Number of users.

    omega : int
        Size of the set of working equations.

    Returns
    -------
    maxiter : 1D ndarray of numpy.uint
        Maximum number of iterations of all Kaczmarz-based RZF detection algos.
        shape: (5,)

    """
    maxiter = np.zeros(4, dtype=np.uint)
    num =  4*K*K*M - 4*K*M + 5*K*K*K + 10*K*K - 2*K

    if omega is None:
        omega = np.ceil(np.log2(K)).astype(np.int)

    # nRK-RZF
    num_nrk = num - K + 1
    den_nrk = 16*M + 8
    maxiter[0] = np.floor(num_nrk/den_nrk).astype(np.uint)

    # RK-RZF
    num_rk = num + 1
    den_rk = K + 16*M + 8
    maxiter[1] = np.floor(num_rk/den_rk).astype(np.uint)

    # rGRK-RZF
    num_rgrk = K*(5*K*K + 11*K - 3)
    den_rgrk = 16*K + 8*M + 7
    maxiter[2] = np.floor(num_rgrk/den_rgrk).astype(np.uint)

    # RSK-RZF
    num_rsk = num
    den_rsk = omega*(8*M + 9) + 8*M + 4
    maxiter[3] = np.floor(num_rsk/den_rsk).astype(np.uint)

    # Search for zeros
    maxiter = np.where(maxiter < 1, 1, maxiter)

    return maxiter
