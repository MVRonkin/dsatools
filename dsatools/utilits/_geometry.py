import numpy as np
from .. import operators

__all__=['calc_line','flip_vector','ols_line','piecewise_ols_line','point_to_vector_distance']

_EPS_  = 1e-4
def _calc_line(stat_point, end_point):
    x1,y1 = stat_point
    x2,y2 = end_point

    if x1 == x2:
        x2 = x1*(1+_EPS_)
        
    b = (x2*y1 - x1*y2)/(x2-x1)
    a = (y2 - y1)/(x2-x1)
   
    return a*np.arange(x1,x2)+b
#----------------------
def calc_line(vector):
    '''
    Calculate Line form vector begining to it end.
    ..math::
        y = a*n+b,
        where:
        a = (vector[N-1]-vector[0])/N
        b = vector[0]
        N = vector.size
        
    Parameters
    ----------
    vector: 1d ndarray,
        input vector

    Returns
    --------
    line: 1d ndarray,
        with the same size as vector.
    '''
    return _calc_line([0,vector[0]], 
                      [len(vector),
                       vector[-1]])
#----------------------
def flip_vector(vector):
    '''
    Flip vector relative to line 
     between its start and end. 
    
    Parameters
    ----------
    vector: 1d ndarray,
        input vector

    Returns
    --------
    fliped vector: 1d ndarray.        
    '''
    line = calc_line(vector)
    return 2*line-vector
#------------------------
def ols_line(vector):
    '''
    Calculate Line approximation of vector
      using the ordinary least-square solution.

    ..math::
      y = a*n+b,
      where:
      a = (N sum(n*vector)-sum(N)sum(vector))/
              (Nsum(n^2)-sum(n)^2)
      b = (sum(vector)-a*sum(n))/N
      N = vector.size

    Parameters
    ----------
    vector: 1d ndarray,
        input vector

    Returns
    --------
    line: 1d ndarray,
        with the same size as vector.
    '''
    vector = np.asarray(vector)
    N = vector.shape[0]
    
    n= np.arange(N)
    sum_n = N*(N-1)/2
    sum_n2 = N*(N-1)*(2*N-1)/6
    
    slope = (N*np.sum(n*vector) -sum_n*np.sum(vector))
    slope /= N*sum_n2-sum_n**2
    
    bias = (np.sum(vector) - slope*sum_n)/N
    
    return slope*n+bias
#------------------------
def piecewise_ols_line(vector, size = 2):
    '''
    Calculate piecewise line approximation of vector
      using the ordinary least-square solution
      for each piece of vector (with predefined size).

    ..math::
      for i in range(stop=vector.size, step=size)  
      y[i:i+size] = a*n+b,
      where:
      a = (N sum(n*vector)-sum(N)sum(vector))/
              (Nsum(n^2)-sum(n)^2)
      b = (sum(vector)-a*sum(n))/N
      N = size

    Parameters
    ----------
    vector: 1d ndarray,
      input vector
    size: int,
      the number of points in each piece.

    Returns
    --------
    piecewise vector: 1d ndarray,
        with the same size as vector.
    '''
    vector = np.asarray(vector)
    N = vector.shape[0]
    size = int(size)
    out = np.zeros_like(vector)
    if size<2:
        return vector
    for i in range(0,N, size):
        lp = min(N,i+size)
        out[i:lp] = ols_line(vector[i:lp])
    return out
#---------------------------
def point_to_vector_distance(vector, p_position, p_value):
    '''
    Calculate Point to Vector distance
    ..math::
      dist = |(x2-x1)(y1-y0)-(x1-x0)(y2-y1)|/d,
        d =  sqrt((x2-x1)^2+(y2-y1)^2),
      where: 
        x1=0,y1 = vector[0]
        x2=vector.size, y2 = vector[-1]
        x0,y0 - point position and value
    
    Note
    -------
    for vector with one point
      ..math::
      dist = sqrt((x1-x0)^2+(y1-y0)^2),
      where x1=0,y1 = vector[0]
    It is assumed that vector is real-valued.  
      
    Parameters
    --------------
    vector: 1d ndarray,
        input vector
    p_position: int,
        position position
    p_value: float,
        point value.
     
    Returns
    -----------
    distance: float,
        distance from vector to value.
    '''
    vec     = np.array(vector).real
    p_position = int(p_position)
    p_value    = float(p_value)
    
    N = vector.size
    if (N < 2):
        return np.sqrt(np.square(p_position) + 
                       np.square(vec[0] - p_value))
    else:
        n = (N-1)*(vec[0] - p_value) + p_position*(N-1 - vec[0])
        d = np.square(N-1) + np.square(vec[-1]-vec[0])
        return np.abs(n) / np.sqrt(d)
