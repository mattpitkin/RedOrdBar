"""
A selection of python functions for using a greedy algorithm to produce a set of
orthonormal functions.

These are based on the functions in the greedycpp code by Scott Field
https://bitbucket.org/sfield83/greedycpp/
"""

import numpy as np

def dot_product(weights, a, b):
    """
    The weighted dot product of two vectors a and b (weights can either be a
    vector or a single value. The vectors a and b can either be real or complex.
    """

    assert len(a) == len(b)
    return np.vdot(a*weights, b)


def mgs(orthobasis, RB, wt):
    """
    The Modified Gram-Schmidt algorithm

    Inputs
    ------
    orthobasis - a numpy vector containing a single orthonormal basis function
    RB - a numpy ndarray containing the current reduced basis
    wt - a single value or vector of weights

    Outputs
    -------
    northobasis - a numpy vector containing the new orthonormal basis
    ru - the projection of the orthonormal basis onto the current reduced basis 
    """
    # get projections on to orthobasis and subtract
    ru = np.zeros(RB.shape[0]+1)
    northobasis = np.copy(orthobasis)

    for i in range(RB.shape[0]):
        basis = np.copy(RB[i])
        L2_proj = dot_product(wt, basis, northobasis)
        ru[i] = L2_proj
        basis *= L2_proj
        northobasis -= basis

    # get normalisation for orthobasis
    norm = np.sqrt(np.abs(dot_product(wt, northobasis, northobasis)))
    ru[RB.shape[0]] = norm
    # normalise orthogonal basis vector
    northobasis /= norm

    return northobasis, ru

def imgs(orthobasis, RB, wt):
    """
    Iterative modified Gram-Schmidt algorithm.
    
    Inputs
    ------
    orthobasis - a numpy vector containing a single orthonormal basis function
    RB - a numpy ndarray containing the current reduced basis
    wt - a single value or vector of weights
    
    Outputs
    -------
    northobasis - a numpy vector containing the new orthonormal basis
    """
    orthocondition = 0.5 # hardcoded condition
    nrm_prev = np.sqrt(np.abs(dot_product(wt, orthobasis, orthobasis)))
    flag = True

    # copy and normalise orthogonal basis vector
    e = np.copy(orthobasis)
    e /= nrm_prev

    ru = np.zeros(RB.shape[0]+1)

    while flag:
        northobasis = np.copy(e) # copy e back into orthobasis
        northobasis, rlast = mgs(northobasis, RB, wt)
        ru += rlast
        nrm_current = ru[-1].real

        northobasis *= ru[-1]

        if  nrm_current/nrm_prev <= orthocondition:
            nrm_prev = nrm_current
            e = np.copy(northobasis)
        else:
            flag = False
        
        nrm_current = np.sqrt(np.abs(dot_product(wt, northobasis, northobasis)))
        northobasis /= nrm_current
        ru[-1] = nrm_current

    return northobasis


def greedy(TS, wt, tol=1e-12):
    """
    Perform greedy algorithm (using IMGS) to form a reduced basis from a given training set.
    
    Inputs
    ------
    TS - a numpy ndarray containing a set of normalised training vectors
    wt - a weight, or vector of weights, for normalisation
    tol - a stopping criterion for the algorithm
    """

    rows = TS.shape[0] # number of training waveforms
    cols = TS.shape[1] # length of each training waveform

    continuework = True

    last_rb = np.zeros(cols)
    ortho_basis = np.zeros(cols)
    errors = np.zeros(rows)
    A_row_norms2 = np.zeros(rows)
    projection_norms2 = np.zeros(rows)
    
    greedy_points = np.zeros(rows)
    greedy_err = np.zeros(rows)
    RB_space = np.zeros((1,cols)) # initialse first reduced basis to zeros

    for i in range(rows):
        # NOT SURE WHY THIS IS SQUARE ROOTED, BUT IT WORKS (SO LONG AS THE TRAINING SET
        # IS ALREADY NORMALISED, SO ALL VALUES ARE UNITY ANYWAY)
        A_row_norms2[i] = np.sqrt(np.abs(dot_product(wt, TS[i], TS[i])))

    # initialise with the first training set
    RB_space[0] = TS[0]

    greedy_points[0] = 0
    dim_RB = 1
    greedy_err[0] = 1.0
    
    while continuework:
        last_rb = RB_space[dim_RB-1] # previous basis

        # Compute overlaps of pieces of TS with rb_new
        for i in range(rows):
            projection_coeff = dot_product(wt, last_rb, TS[i])
            projection_norms2[i] += projection_coeff.real**2 + projection_coeff.imag**2
            errors[i] = A_row_norms2[i] - projection_norms2[i]

        # find worst represented TS element and add to basis
        i = np.argmax(errors)
        worst_err = errors[i]
        worst_app = i

        greedy_points[dim_RB] = worst_app
        greedy_err[dim_RB] = worst_err

        print dim_RB, worst_err, worst_app

        orthobasis = np.copy(TS[worst_app])
        orthobasis = imgs(orthobasis, RB_space, wt) # use IMGS
        RB_space = np.vstack((RB_space, orthobasis))

        dim_RB += 1

        # decide if another greedy sweep is needed
        if worst_err < tol or rows == dim_RB:
            continuework = False
 
    return RB_space