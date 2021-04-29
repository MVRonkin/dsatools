import numpy as np

__all__=['barycenter',]

def barycenter(vector, n_range=None):
    '''
    Barycenter calculation as
    ..math::
    sum(n*vector)/sum(n)

    where n is range n_range[0],..., n_range[1]
    
    Parameters
    -------------
    * vector: 1d ndarray,
        input vector.
    * n_range: [int,int]; int or None
        > if[int,int] is first and last points  
            in range of calculation;
        > if int is [0,int];
        > if None is [0,vector.size];
    Returns
    -----------
    * float,
        barycenter value.    
    '''
    vector = np.asarray(vector)
    if n_range is None: n_range = [0,vector.size]
    
    n_range = np.asarray(n_range)
    
    if n_range.size==1: n_range = np.append([0],n_range) 
    if n_range.size>1:  n_range = n_range[:2]
    
    if np.max(n_range)>vector.size: n_range[1] = vector.size
    if np.max(n_range)>vector.size: n_range[1] = vector.size
        
    n = np.arange(vector.size)
    
    n = n[n_range[0]:n_range[1]]
    vector = vector[n_range[0]:n_range[1]]
    
    return np.sum(vector*n)/np.sum(n)