import numpy as np
from ... import operators 

__all__ = ['hasudorf','hasudorf_test','dtw','ddtw','msm','erp','lcss','twe']

_EPS_ = 1e-7
# ----------------------------------------------------------------------------------------
def _check_xy(x, y=None):
    x = np.asarray(x)
    if y is None:
        y = x
    else:
        y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError('x.shape != y.shape')

    return x, y
# ----------------------------------------------------------------------------------------
def hasudorf(x,y, square=True, normalize=True):
    '''
    Hasudorf distance: 
    ..math::
      D = sup(inf_yi dist(yi,X) - inf_xi dist(xi,Y))
    where: dist(x,Y) = sum((Y-x)^2)
    
    Parameters
    --------------
    * x, y: 1d ndarrays,     
    * square: bool,
        if false, than sqrt will be taken.
    * normalize: bool,
        if true, distance will be 
        normalized as dist = dist/(std(x)*std(y)) 
        
    Returns:
    --------
    * d, float:
        distance value.
        
    '''
    x,y = _check_xy(x, y)
    D = operators.euclidian_matrix(x,y, 
                                   inner=False, 
                                   square=square, 
                                   normalize=normalize) 
    #TODO: i'm not sure weather it is need here sum or not
    inf_y_X = np.min(np.sum(D,axis=0)) 
    inf_X_u = np.min(np.sum(D,axis=1))

    return np.max(inf_y_X-inf_X_u)

#------------------------------------
def hasudorf_test(x,y, square=True, normalize=True, reduction = None):
    '''
    TEST Hasudorf distance: 
    ..math::
      D = sup(inf_yi dist(yi,X) - inf_xi dist(xi,Y))
    where: dist(x,Y) = sum((Y-x)^2)
    
    Parameters
    --------------
    * x, y: 1d ndarrays,     
    * square: bool,
        if false, than sqrt will be taken.
    * normalize: bool,
        if true, distance will be 
        normalized as dist = dist/(std(x)*std(y)) 
        
    Returns:
    --------
    * d, float:
        distance value.
        
    '''
    x,y = _check_xy(x, y)
    D = operators.euclidian_matrix(x,y, 
                                   inner=False, 
                                   square=square, 
                                   normalize=normalize) 
    #TODO: i'm not sure weather it is need here sum or not
    
    if reduction =='sum':
        inf_y_X = np.min(np.sum(D,axis=0)) 
        inf_X_u = np.min(np.sum(D,axis=1))
    elif reduction =='max':
        inf_y_X = np.min(np.max(D,axis=0)) 
        inf_X_u = np.min(np.max(D,axis=1))
    elif reduction ==None:
        inf_y_X = np.min(D,axis=0)
        inf_X_u = np.min(D,axis=1)        
    

    return np.max(inf_y_X-inf_X_u)

#------------------------------------
def dtw( x,  y, weight = None ):
    '''
    Dinamic Time Wrapping Distance,
      Algorithm, based on the searching of 
      maximum similarity between points 
      independently of its index position.
    
    ..math::    
    D(i,j) =dist(x_i,y_j)
    D(i,j) += min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
    d_{DWT} = min_i sum_j=0^K D(i,j)/K
    where dist(x,y) = (x-y)^2 
    
    Parameters
    --------------
    * x, y: 1d ndarrays,
    * weight: value of weights in the form:
      ..math::
        1/(1+np.exp(-weight*(np.arange(N)-N//2)))
        where N is the range(0,x.shape[0])
   
   Returns:
    --------
    * d, float:
        distance value.     
    
    Note:
    ---------
    Algorithm:
    1. Calculate the distance between 
      the first point in the first signal 
      and every point in the second series.
    2. Select the minimum of the calculated 
      values and store it 
      (this is the "time warp" stage).
    3. Move to the second point and repeat stage 1.
    4. Move step by step along points and repeat stage 1 
       till all points are exhausted.
    5. Calculate and Select the minimum of distances 
       between the first point in the second series 
       segment and every point in the first series.
    6. Move step by step along points in the second 
       segment and repeat stage 3 till all 
       points are exhausted.
    7. Sum all the stored minimum distances.
    
    '''
    x,y = _check_xy(x, y)
    
    N = x.shape[0]

    n_rows    = N + 1
    n_columns = N + 1
           
    D = np.zeros((n_rows,n_columns))

    D[0,1:] = np.inf
    D[1:,0] = np.inf

    D[1:,1:] = operators.euclidian_matrix(x,y) 
    
    if weight is not None:
        weight_vector = 1/(1+np.exp(-weight*(np.arange(N)-N//2)))
        
        #TODO: simplify
        for row in range(1,n_rows):
            for column in range(1,n_columns):
                D[row,column] *= weight_vector[abs(row-column)]


    for i in range(1,n_rows):
        for j in range(max(1, i-N),n_columns):
            D[i,j] = D[i,j] + min(D[i-1,j],D[i,j-1],D[i-1,j-1])

    return D[N,N]
#---------------------------
def ddtw(x,  y, weight = None):
    '''
    Dinamic Time Wrapping Distance of 
    derevatives,
      Algorithm, based on the searching of 
      maximum similarity between points 
      independently of its index position.
    
    ..math::    
    D(i,j) =dist(x_i,y_j)
    D(i,j) += min{D(i-1,j), D(i,j-1), D(i-1,j-1)}
    d_{DWT} = min_i sum_j=0^K D(i,j)/K
    where dist(x,y) = (diff(x)-diff(y))^2 
    
    Parameters
    --------------
    * x, y: 1d ndarrays,
    * weight: value of weights in the form:
      ..math::
        1/(1+np.exp(-weight*(np.arange(N)-N//2)))
        where N is the range(0,x.shape[0])
   
   Returns:
    --------
    * d, float:
        distance value.     
    
    Note:
    ---------
    Algorithm:
    1. Calculate the distance between 
      the first point in the first signal 
      and every point in the second series.
    2. Select the minimum of the calculated 
      values and store it 
      (this is the "time warp" stage).
    3. Move to the second point and repeat stage 1.
    4. Move step by step along points and repeat stage 1 
       till all points are exhausted.
    5. Calculate and Select the minimum of distances 
       between the first point in the second series 
       segment and every point in the first series.
    6. Move step by step along points in the second 
       segment and repeat stage 3 till all 
       points are exhausted.
    7. Sum all the stored minimum distances.
    
    '''
    return dtw(np.diff(x), np.diff(y))
#---------------------------
def msm(x, y, c = 0.01):
    ''' TEST '''
    N = x.shape[0]

    cost = np.zeros((N,N))

    # Initialization
    cost[0, 0] = abs(x[0] - y[0])

    for i in range(1, N):
        cost[i,0] = cost[i - 1,0] + _msm_cost(x[i], x[i - 1], y[0],c)
        cost[0,i] = cost[0,i - 1] + _msm_cost(y[i], x[0], y[i - 1],c)

    for i in range(0, N):
        for j in range(0, N):
            d1 = cost[i - 1, j - 1] + abs(x[i] - y[j])            
            d2 = cost[i - 1, j] + _msm_cost(x[i], x[i - 1], y[j], c)            
            d3 = cost[i, j - 1] + _msm_cost(y[j], x[i], y[j - 1], c)
            
            cost[i, j] = min(d1,d2,d3)

    return cost[N - 1, N - 1]

#---------------------------
def _msm_cost(new_point, x, y, c):
    ''' TEST '''
    if ((x <= new_point) and (new_point <= y)) or ((y <= new_point) and (new_point <= x)):
    
#     if x<=new_point<=y or y<=new_point<=x:
        return c
    else:
        return c + min(abs(new_point - x), abs(new_point - y))
#---------------------------
def lcss(x, y, delta= 3, epsilon = 0.05):
    ''' TEST '''
    N = x.shape[0]

    lcss = np.zeros((N + 1, N + 1), dtype=np.int32)

    for i in range(N):
        for j in range(i - int(delta), i + int(delta) + 1):
            
            if j < 0:  j = -1
            
            elif j >= N: j = i + delta
            
            elif (y[j].real + epsilon >= 
                  x[i].real >= 
                  y[j].real - epsilon):
                
                lcss[i + 1, j + 1] = lcss[i,j] + 1
            
            elif lcss[i,j + 1] > lcss[i + 1,j]:
                lcss[i + 1,j + 1] = lcss[i,j + 1]
            
            else:
                lcss[i + 1,j + 1] = lcss[i + 1, j]

    max_val = -1
    
    for i in range(1, N+1):
        if lcss[N, i] > max_val:
            max_val = lcss[N, i]

    return 1 - (max_val / N)   

#---------------------------
def erp(x, y,  band_size = 5, g = 0.5):
    ''' TEST '''
    N = x.shape[0]

    band = np.ceil(band_size * N)

    curr = np.empty(N)
    prev = np.empty(N)

    for i in range(0, N):

        prev,curr = curr, prev

        l = max(0,i - int(band) -1)           
        r = min(i + int(band) + 1,N - 1)       

        for j in range(l, r + 1):
            if np.abs(i - j) <= int(band):

                dist1  = abs(x[i] - g)**2
                dist2  = abs(g - y[j])**2
                dist12 = abs(x[i] - y[i])**2

                if i + j != 0:
                    if i == 0 or (j != 0 and 
                                  (
                                   ((prev[j - 1] + dist12) > (curr[j - 1] + dist2)) and 
                                   ((curr[j - 1] + dist2) < (prev[j] + dist1))
                                  )
                                 ):
                        # del
                        cost = curr[j - 1] + dist2
                    
                    elif (j == 0) or ((i != 0) and 
                                      (
                                       ((prev[j - 1] + dist12) > (prev[j] + dist1)) and 
                                       ((prev[j] + dist1) < (curr[j - 1] + dist2))
                                      )
                                     ):
                        # ins
                        cost = prev[j] + dist1;
                    
                    else:
                        # match
                        cost = prev[j - 1] + dist12
                
                else:
                    cost = 0

                curr[j] = cost

    return np.sqrt(curr[N - 1])

#--------------------------------
def twe(ta, tb, penalty = 1, stiffness = 1):
    ''' TEST '''
    r = len(ta)
    c = len(tb)
    
    tsa = np.zeros(len(ta) + 1)
    tsb = np.zeros(len(tb) + 1)

    dim = 0
    for i in range(0, len(tsa)):
        tsa[i] = (i + 1)
    for i in range(0, len(tsb)):
        tsb[i] = (i + 1)
        
    D = np.zeros((r + 1, c + 1))
    Di1 = np.zeros(r + 1)
    Dj1 = np.zeros(c + 1)     

    # local costs initializations
    for j in range(1, c + 1):
        distj1 = 0
        for k in range(0, dim + 1):
            if j > 1:
                distj1 += abs(tb[j - 2] - tb[j - 1])**2
            else:
                distj1 += abs(tb[j - 1])**2
        Dj1[j] = distj1

    for i in range(1, r + 1):
        disti1 = 0
        for k in range(0, dim + 1):
            if i > 1:
                disti1 += abs(ta[i - 2] - ta[i - 1])**2
            else:
                disti1 += abs(ta[i - 1])**2
        Di1[i] = disti1

        for j in range(1, c + 1):
            dist = 0
            for k in range(0, dim + 1):
                dist += abs(ta[i - 1] - tb[j - 1])**2
                if i > 1 and j > 1:
                    dist += abs(ta[i - 2] - tb[j - 2])**2
            D[i,j] = dist

    # border of the cost matrix initialization
    D[0,0] = 0
    for i in range(1, r + 1):
        D[i,0] = D[i - 1,0] + Di1[i]
    for j in range(1, c + 1):
        D[0,j] = D[0,j - 1] + Dj1[j]

    for i in range(1, r + 1):
        for j in range(1, c + 1):
            
            htrans = abs(tsa[i - 1] - tsb[j - 1])
            
            if j > 1 and i > 1: htrans += abs(tsa[i - 2] - tsb[j - 2])
            
            dist0 = D[i - 1,j - 1] + stiffness * htrans + D[i,j]
            dmin = dist0
            
            if i > 1: htrans = tsa[i - 1] - tsa[i - 2]
            else: htrans = tsa[i - 1]
            
            dist = Di1[i] + D[i - 1,j] + penalty + stiffness * htrans            
            if dmin > dist: dmin = dist
            
            if j > 1: htrans = tsb[j - 1] - tsb[j - 2]
            else: htrans = tsb[j - 1]
                
            dist = Dj1[j] + D[i,j - 1] + penalty + stiffness * htrans
            if dmin > dist: dmin = dist
            
            D[i,j] = dmin

    dist = D[r,c]
    return dist