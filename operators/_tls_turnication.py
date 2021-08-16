import numpy as np
import scipy

#-------------------------------------------------
def tls_turnication(X, Y, tls_rank=0):
    ''' 
    Total Least Square turnication.
    
    Parameters
    --------------
    * X,Y: 2d ndarrays,
        input matrices.
    * tls_rank: int,
        rank for turnication.
    
    Returns
    ----------
    * X_new, Y_new: 2d ndarrays, 
        turnicated matrices.
    
    
    References:
    ------------
    [1] https://github.com/mathLab/PyDMD/blob/master/pydmd    
    '''
    if tls_rank in [0,None]: return X, Y
    
    _,_,Vh = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)
    
    Vh  = Vh[: np.min([tls_rank, Vh.shape[0]]), :]

    VV    = np.dot(np.conj(Vh.T),Vh)
    
    return X.dot(VV), Y.dot(VV) 

#--------------------------------------------------
def _tls_(X, Y,tls_rank=0):
    tls_rank = 511
    _,_,Vh = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)
    Vxy     = Vh[:tls_rank, tls_rank:]
    Vxx     = Vh[tls_rank:, tls_rank:]
    print(Vxy.shape,Vxx.shape,X.shape)
    QZ = - np.dot(Vxx,np.linalg.pinv(Vxy))

    return QZ

#--------------------------------------------------
def _tlsV2_(X,y):
    if len(X.shape) is 1:
        n = 1
        X = X.reshape(len(X),1)
    else:
        n = np.array(X).shape[1] # the number of variable of X
    
    Z = np.vstack((X.T,y)).T
    U, s, Vt = la.svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy  / Vyy # total least squares soln
    
    
#     Xtyt = - Z.dot(V[:,n:]).dot(V[:,n:].T)
#     Xt = Xtyt[:,:n] # X error
#     y_tls = (X+Xt).dot(a_tls)

#     fro_norm = la.norm(Xtyt, 'fro')#Frobenius norm
    
    return a_tls#y_tls, X + Xt, a_tls, fro_norm   


#--------------------------------------------------
def linear_regression(X, y):
    """Return the weights from linear regression.
    X: nxd (d = 1) matrix of data organized in rows
    y: length n vector of labels
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

