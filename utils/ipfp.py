"""
IPFP (Iterative Proportional Fitting Procedure) Algorithm
and Partial Information Decomposition (PID) utilities.
"""

import numpy as np
from scipy.special import logsumexp


def alternating_minimization_ipfp(P, rng_seed=42, max_outer=50, max_sink=100,
                                  tol_outer=1e-8, tol_sink=1e-8, eps=1e-20, 
                                  verbose=False):
    """
    Alternating minimization IPFP algorithm for computing PID.
    
    Args:
        P: Joint distribution P(X1, X2, Y)
        rng_seed: Random seed for reproducibility
        max_outer: Maximum outer iterations
        max_sink: Maximum Sinkhorn iterations
        tol_outer: Tolerance for outer convergence
        tol_sink: Tolerance for Sinkhorn convergence
        eps: Small constant to avoid log(0)
        verbose: Print iteration details
        
    Returns:
        Q: Optimized distribution for redundancy computation
    """
    np.random.seed(rng_seed)
    
    # Compute marginals
    Px1y = P.sum(axis=1)
    Px2y = P.sum(axis=0)
    py = Px1y.sum(axis=0)

    if not np.allclose(py, Px2y.sum(axis=0), atol=1e-8):
        raise ValueError("Marginals Px1y and Px2y are not consistent with py.")

    m, k = Px1y.shape
    n, _ = Px2y.shape

    # Initialize Q
    Q = np.zeros((m, n, k))
    for y in range(k):
        if py[y] > eps:
            Q[:, :, y] = np.outer(Px1y[:, y], Px2y[:, y]) / py[y]
    Q = np.maximum(Q, eps)
    Q /= Q.sum()
    prev_obj = np.inf

    # Alternating minimization
    for t in range(max_outer):
        Q_marg = Q.sum(axis=2)
        A = Q_marg / k
        A_log = np.log(np.maximum(A, eps))
        Q_new = np.zeros_like(Q)

        # Sinkhorn iterations for each Y value
        for y in range(k):
            if py[y] < eps:
                continue
            log_r = np.log(np.maximum(Px1y[:, y], eps))
            log_c = np.log(np.maximum(Px2y[:, y], eps))
            log_v = np.zeros(n)
            
            for s in range(max_sink):
                log_u = log_r - logsumexp(A_log + log_v[np.newaxis, :], axis=1)
                log_v_new = log_c - logsumexp(A_log + log_u[:, np.newaxis], axis=0)

                if s > 0 and np.max(np.abs(log_v_new - log_v)) < tol_sink:
                    break
                log_v = log_v_new
                
            Q_new[:, :, y] = np.exp(A_log + log_u[:, np.newaxis] + log_v[np.newaxis, :])
            
        Q_new = np.maximum(Q_new, eps)
        Q_new /= Q_new.sum()

        # Compute objective
        Q_marg_new = Q_new.sum(axis=2)
        Q_tilde = Q_marg_new[:, :, None] / k
        obj = np.sum(Q_new * (np.log(Q_new + eps) - np.log(Q_tilde + eps)))

        if verbose:
            print(f"Iteration {t+1}, objective = {obj:.6e}")

        # Check convergence
        if t > 0 and np.abs(prev_obj - obj) / max(1.0, np.abs(prev_obj)) < tol_outer:
            break

        Q = Q_new
        prev_obj = obj

    return Q


def convert_data_to_distribution(x1, x2, y):
    """
    Convert discrete data arrays to joint probability distribution.
    
    Args:
        x1: First input variable (flattened array)
        x2: Second input variable (flattened array)
        y: Output variable (flattened array)
        
    Returns:
        joint_distribution: P(X1, X2, Y)
        maps: Tuple of dictionaries mapping raw values to discrete indices
    """
    assert x1.size == x2.size
    assert x1.size == y.size
    numel = x1.size

    x1_discrete, x1_raw_to_discrete = extract_categorical_from_data(x1.squeeze())
    x2_discrete, x2_raw_to_discrete = extract_categorical_from_data(x2.squeeze())
    y_discrete, y_raw_to_discrete = extract_categorical_from_data(y.squeeze())

    joint_distribution = np.zeros((
        len(x1_raw_to_discrete), 
        len(x2_raw_to_discrete), 
        len(y_raw_to_discrete)
    ))
    
    for i in range(numel):
        joint_distribution[x1_discrete[i], x2_discrete[i], y_discrete[i]] += 1
        
    joint_distribution /= np.sum(joint_distribution)

    return joint_distribution, (x1_raw_to_discrete, x2_raw_to_discrete, y_raw_to_discrete)


def extract_categorical_from_data(x):
    """
    Convert continuous/arbitrary data to categorical indices.
    
    Args:
        x: Input array
        
    Returns:
        discrete_data: List of discrete indices
        raw_to_discrete: Dictionary mapping original values to indices
    """
    supp = set(x)
    raw_to_discrete = dict()
    for i in supp:
        raw_to_discrete[i] = len(raw_to_discrete)
    discrete_data = [raw_to_discrete[x_] for x_ in x]
    return discrete_data, raw_to_discrete


def get_measure(P, name="ipfp", max_iters=500):
    """
    Compute PID measures (redundancy, unique information, synergy).
    
    Args:
        P: Joint distribution P(X1, X2, Y)
        name: Method name (currently only 'ipfp' supported)
        max_iters: Maximum IPFP iterations
        
    Returns:
        Dictionary with 'redundancy', 'unique1', 'unique2', 'synergy'
    """
    if name == 'ipfp':
        Q = alternating_minimization_ipfp(P, max_outer=max_iters)
    else:
        raise ValueError(f"Unknown method: {name}")

    from .metrics import CoI, UI, CI
    
    redundancy = CoI(Q)
    unique1 = UI(Q, cond_id=1)  # Unique to X1 (text)
    unique2 = UI(Q, cond_id=0)  # Unique to X2 (image)
    synergy = CI(P, Q)

    return {
        'redundancy': redundancy, 
        'unique1': unique1, 
        'unique2': unique2, 
        'synergy': synergy
    }
