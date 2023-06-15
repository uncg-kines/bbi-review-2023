# 
# nate berry
# 2023-04-01
#
# ---------------------------------------------------------

print(''' 
      \n\n
      # -------------------------------------------------
      author: nate berry (nberry11 at gmail dot com)
      citation: 
        <tbd>
      # -------------------------------------------------
      this script contains a variety of functions used to
      quantify the dynamics and complexity of a time series

      specific functions called from this script include:
            < acf >   autocorrelation function
            < ami >   average mutual information
            < Amat >  Takens' reconstruction
            < fnn >   false nearest neighbors
      
      other required packages (imported within the working 
      scripts) include: 
            < EntropyHub >  sample entropy & cross sample entropy
            < pyrqa >       RQA and cRQA
      
      we have worked diligently to make sure that there are
      no errors within this code and that everything is 
      reproducible, without error. if you come across an error,
      please let us know.

      # -------------------------------------------------
      \n\n 
      ''')


import numpy as np 
import pandas as pd 

from sklearn.feature_selection import mutual_info_regression


def acf(ts, maxlags=20):
    """
    ----------------------------------------------------------
    Autocorrelation function
    ----------------------------------------------------------
    NOTES: 
    - provides an easy plotting output
    - requires numpy array as <ts>
    ----------------------------------------------------------
    """
    var = np.var(ts)
    x = ts-ts.mean()

    acorr = np.correlate(x, x, 'full')[len(x)-1:] 
    acorr = acorr / var / len(x)

    return [np.arange(0, maxlags), acorr[:maxlags]]



def Amat(ts, m, tau):
    """
    ----------------------------------------------------------
    Create the trajectory matrix
    ----------------------------------------------------------
    """
    N = len(ts)-m*tau
    A = np.zeros([N, m])  # initiate A 
    for j in range(0, m):
        A[:, j] = ts[j*tau: len(ts)-tau*(m-j)]

    return A



def ami(ts, maxlags=128):
    """
    ----------------------------------------------------------
    Average mutual information calculation
    ----------------------------------------------------------
    NOTES: 
    - utilizing sklearn feature selection mutual_info_regression
    ----------------------------------------------------------
    """
    ts = np.array(ts)
    ts = pd.DataFrame(ts.reshape(len(ts), 1))
    d = np.zeros([len(ts), maxlags])

    for L in range(maxlags):
        d[:, L] = ts.shift(L).to_numpy().reshape(-1)

    f = np.isnan(d)
    d[f] = 0
    y, X = d[:, 0], d[:, 1:]

    mi = mutual_info_regression(X, y)

    return [np.arange(1, maxlags), mi]



def cross_ami(ts1, ts2, maxlags=64):
    """
    ----------------------------------------------------------
    Cross-average mutual information calculation
    ----------------------------------------------------------
    NOTES: 
    - utilizing sklearn feature selection mutual_info_regression
    ----------------------------------------------------------
    """
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    
    ts1 = pd.DataFrame(ts1.reshape(len(ts1), 1))
    ts2 = pd.DataFrame(ts2.reshape(len(ts2), 1))

    d1 = np.zeros([len(ts1), maxlags])
    d2 = np.zeros([len(ts2), maxlags])

    for L in range(maxlags):
        d1[:, L] = ts1.shift(L).to_numpy().reshape(-1)
        d2[:, L] = ts2.shift(L).to_numpy().reshape(-1)

    f = np.isnan(d1)
    d1[f] = 0
    d2[f] = 0

    y1, X1 = d1[:, 0], d1[:, :]
    y2, X2 = d2[:, 0], d2[:, :]
    
    # swap X1 and X2 with y1 and y2 for crossed time series
    mi1 = mutual_info_regression(X2, y1)
    mi2 = mutual_info_regression(X1, y2)

    return [np.arange(0, maxlags), mi1, mi2]



def fnn(A, tol=20):
    """
    ----------------------------------------------------------
    False nearest neighbors 
    code adapted from Shelhamer (2006) "NLD in Physiology"
    ----------------------------------------------------------
    NOTES: 
    - calculate the proportion of false nearest neighbors 
      in the attractor over a range of embedding dimensions 
    - provies analysis for determination of embedding dimension 
    ----------------------------------------------------------
    """
    # tol = np.std(A[:,0]) * tol
    N, max_m = A.shape
    per = np.zeros(max_m)

    # loop through embeddings
    for m in range(1, max_m-1):
        # loop through/pick each reference 
        for ir in np.arange(N):
            ref = A[ir, :]
            # compare all other points to reference 
            dst = np.inf
            for ii in range(N):
                dd = 0
                for k in range(1, m+1):
                    dd = dd + (A[ir, k] - A[ii, k])**2

                dd = np.sqrt(dd)
                if (ii != ir) & (dd < dst):
                    dst, inear = dd, ii

            # calculate dist at next m
            dst2 = np.sqrt(dst**2 + (A[ir, m+1] - A[inear, m+1])**2)
            if (dst2/dst > tol):
                per[m-1] = per[m-1]+1

    per = per / (N+1-1)
    
    return [np.arange(1, max_m+1), per]


# end 
