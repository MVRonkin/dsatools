import numpy as np
import scipy

#-----------------------------------
def teager(x):
    '''
    Teager operator.
    
    Parameters
    ---------
    * x: 1d ndarray,
        input signal.
    
    Returns
    --------
    * xt: transformed signal.
    
    Notes
    -------
    * Teager operator has the following form:
        x^2+x[1:]x[-1]
    '''
    return np.append([0,0],x[1:-1]*x[1:-1] - x[2:]*x[:-2])
