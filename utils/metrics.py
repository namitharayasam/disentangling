"""
Information-theoretic metrics for PID analysis.
"""

import numpy as np
from scipy.special import rel_entr


def MI(P):
    """
    Compute Mutual Information I(X; Y) from joint distribution P(X, Y).
    
    Args:
        P: Joint probability distribution (2D array)
        
    Returns:
        Mutual information value
    """
    margin_1 = P.sum(axis=1)
    margin_2 = P.sum(axis=0)
    outer = np.outer(margin_1, margin_2)
    return np.sum(rel_entr(P, outer))


def CoI(P):
    """
    Compute Co-Information (Interaction Information).
    
    CoI(X1; X2; Y) = I(X1; X2) + I(X1; Y) + I(X2; Y) - I(X1, X2; Y)
    
    Args:
        P: Joint distribution P(X1, X2, Y) (3D array)
        
    Returns:
        Co-information value
    """
    A = P.sum(axis=1)  # P(X1, Y)
    B = P.sum(axis=0)  # P(X2, Y)
    C = P.transpose([2, 0, 1]).reshape((P.shape[2], -1))  # P(Y, X1X2)
    return MI(A) + MI(B) - MI(C)


def CI(P, Q):
    """
    Compute Conditional Information (difference between mutual informations).
    
    Args:
        P: Original joint distribution P(X1, X2, Y)
        Q: Optimized distribution Q(X1, X2, Y)
        
    Returns:
        Conditional information value
    """
    assert P.shape == Q.shape
    P_ = P.transpose([2, 0, 1]).reshape((P.shape[2], -1))
    Q_ = Q.transpose([2, 0, 1]).reshape((Q.shape[2], -1))
    return MI(P_) - MI(Q_)


def UI(P, cond_id=0):
    """
    Compute Unique Information conditioned on one variable.
    
    Args:
        P: Joint distribution P(X1, X2, Y)
        cond_id: 0 for unique to X2 (image), 1 for unique to X1 (text)
        
    Returns:
        Unique information value
    """
    if cond_id == 0:
        # Unique to X2: I(X2; Y | X1)
        J = P.sum(axis=(1, 2))
        s = 0
        for i in range(P.shape[0]):
            p = P[i, :, :] / P[i, :, :].sum()
            s += MI(p) * J[i]
        return s
    elif cond_id == 1:
        # Unique to X1: I(X1; Y | X2)
        J = P.sum(axis=(0, 2))
        s = 0
        for i in range(P.shape[1]):
            p = P[:, i, :] / P[:, i, :].sum()
            s += MI(p) * J[i]
        return s
    else:
        raise ValueError("cond_id must be 0 or 1")
