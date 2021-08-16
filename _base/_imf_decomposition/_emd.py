import numpy as np
import scipy  

import scipy.interpolate #import Akima1DInterpolator, Rbf, InterpolatedUnivariateSpline, BSpline


def emd(x, order,method = 'cubic', max_itter = 100, tol = 0.1):
    '''
    Emperical Mode Decomposition (EMD).
    
    The emperical mode deocomposition method is the nonlinear 
       time domain decomposition on the so-called 
       intrinsic mode functions (IMF), based on the idea, 
       that each component can be reconstructed by it envelope.
    
    Parameters
    ----------------
    * x: 1d ndarray.
    * order: int,
        number of IMFs (with out remainder).
    * method: string, 
        method of spline approximation: 
        method = {cubic, akim, rbf, linear, thin_plate}.
    * max_itter: int,
        maximum number of itteration to search imf.
    * tol: float,
        tolerance to variance of changing imf in itterations.
    
    Returns
    ---------------
    * imfs: 2d ndarray,
        intrinsic mode functions and remainder, 
        shape = (order+1,x.shape).

    References
    -----------------
    [1] N. E. Huang et al., 
        "The empirical mode decomposition and the Hilbert 
       spectrum for nonlinear and non-stationary time series analysis", 
       Proc. R. Soc. Lond. A, Math. Phys. Sci., 
       vol. 454, no. 1971, 903–995, (1998).
    [2] N. E. Huang, 
        "Hilbert-Huang transform and its applications", 
        vol. 16. World Scientific, 2014.
    [3] Z. Wu, N. E. Huang, 
        "Ensemble empirical mode decomposition: 
        A noise-assisted data analysis method", 
        Adv. Adapt. Data Anal., vol. 1, no. 1, 1–41 (2008).
    [4] J. Zheng, J. Cheng, Y. Yang, 
        "Partly ensemble empirical mode decomposition: 
        An improved noise-assisted method for eliminating mode mixing", 
        Signal Process., vol. 96, 362–374, (2014).

    
    See also
    -----------------------
    vmd
    hvd
    ewt
    hht (operators)
    
    '''
    
    x = np.array(x)
    N = x.shape[0]
    imf = np.zeros((order, N),dtype = x.dtype)
    
    iscomplex = False
    if x.dtype in [complex,np.complex,np.complex64,np.complex128]:
        iscomplex = True
    
    for ord_cnt in range(order):
        h = x

        for cnt in range(max_itter):
            
            s1 =  get_envelope(h,  method = method)
            s2 = -get_envelope(-h, method = method)
            
            mean_env = (s1+s2)/2
            
            # for RBF interpolation envelope is complex
            if iscomplex and mean_env.dtype \
                not in [complex,
                        np.complex,
                        np.complex64,
                        np.complex128]:
                
                h = h - scipy.signal.hilbert(mean_env)
            
            else:
                h = h - mean_env
            
            #Cashy Criteria  
            sd = np.sum(np.square(mean_env))/np.sum(np.square(h))

            if (np.abs(sd) < tol) or isimf(h):
                break

        imf[ord_cnt,:] = h
        x = x-h
        if ismonotonic(x):
            break

    return imf  

#--------------------------------------------------------
def ismonotonic(x):
    '''
    if there are exists maximums and minimums, False.
    '''
    pmax=findpeaks(x)
    pmin=findpeaks(-x)

    if pmax.size*pmin.size > 0:
        return False 
    else:      
        return True

#--------------------------------------------------------
def isimf(x):
    '''
    if |zero crossing - extremums| less or equal to 1, than IMF
    '''

    N  = x.shape[0];

    # zero crossing
    df = (x[1:]*x[:-1])
    zc = np.sum(df[df<0])
    
    pmax=findpeaks(x)
    pmin=findpeaks(-x)
    extremums = pmax.size+pmin.size
    
    if abs(zc-extremums) > 1: 
        return False
    else:              
        return True

#--------------------------------------------------------
def get_envelope(x, method = 'cubic'):
    '''
    Function to estimate envelope by spline method.
    
    '''
    N = x.shape[0];
    p = findpeaks(x)
    
    if(p.size<2):
        return np.zeros(N)
    
    points = np.concatenate([[0], p, [N]])
    values = np.concatenate([[0], x[p], [0]])
    #TODO check for mirror extention in  my experiments it was worse
#     values, points = x[p],p
#     values,points =_extension(values, points, n_points=2)
    
    new_points = np.arange(points[0],points[-1])
    
    fp = np.flatnonzero(new_points == 0)[0]
    
    s=_spline(values, points, new_points, method = method)[fp:fp+N]
    
    return s

#--------------------------------------------------------
def _spline(values, points, new_points, method = 'cubic'):
    '''
    scipy.interpolate methods.
    '''
    if(method=='cubic'):
        cofs = scipy.interpolate.splrep(points, values)
        return scipy.interpolate.splev(new_points, cofs)
    elif(method=='akim'):
        return scipy.interpolate.Akima1DInterpolator(points,values)(new_points)
    elif(method=='rbf'):
        return scipy.interpolate.Rbf(points,values, function='gaussian')(new_points)
    elif(method=='thin_plate'):
        return scipy.interpolate.Rbf(points,values, function='thin_plate')(new_points) 
    elif(method=='linear'):
        return scipy.interpolate.Rbf(points,values, function='linear')(new_points)   
#--------------------------------------------------------
def findpeaks(x):
    ''' find maximums of signals.
    '''
    return scipy.signal.argrelmax(np.real(x))[0]
#--------------------------------------------------------
def _extension(values, points, n_points=2,mirror = True ): 
    '''
    Mirror extention
    FOR TEST
    '''
    N = values.shape[0]
    
    if mirror:        
        values = np.concatenate(( values[n_points-1::-1], 
                                 values, 
                                 values[N-1:N-n_points-1:-1] ))
    else:

        values = np.concatenate(( values[n_points:0:-1], 
                                 values, 
                                 values[N-2:N-n_points-2:-1] ))
        
    points = np.concatenate((2*points[0] - points[n_points:0:-1], 
                            points, 
                            2*points[-1] - points[N-2:N-n_points-2:-1]))
    return values, points
    

# __all__ = ['emd_filter','emd']
# #--------------------------------------------------------------------
# _MIN_EXTREMUMS = 4 #Requirement of scipy
# TOL = 0.00005 # determined emperically
# #--------------------------------------------------------------------
# def emd_filter(x, method = 'cubic', max_itter=1):
#     '''
#     Emperical Mode Decomposition (EMD) filter.
#     The filter based on the serching for first 
#     intrinsic mode function and subtract it.
    
#     Parameters:
#     --------------------------------------------
#     * x: input 1d ndarray.
#     * order:  number of IMFs (with out remainder).
#     * method: method of spline approximation: {cubic, akim, rbf, linear, thin_plate}.
#     * max_itter: maximum number of itteration to search imf.
    
#     Returns:
#     -------------------------------------------
#     * filtered signal.
#     '''
#     out = np.array(x)
#     for _ in np.arange(max_itter):
#         envdw, envup, _ = _envelops(out, method = method)
#         out -= 0.5*(envdw+envup)
#     return out

# #--------------------------------------------------------------------
# def emd(x, order=None,  method = 'cubic', max_itter=100):
#     '''
#     Emperical Mode Decomposition (EMD).
    
#     The emperical mode deocomposition method is the nonlinear time
#        domain decomposition on the so-called intrinsic mode functions (IMF), 
#        based on the idea, that ech component can be reconstructed by searching it envelope.
    
#     Parameters:
#     ---------------------------------------------------------
#     * x: input 1d ndarray.
#     * order:  number of IMFs (with out remainder).
#     * method: method of spline approximation: {cubic, akim, rbf, linear, thin_plate}.
#     * max_itter: maximum number of itteration to search imf.
    
#     Returns:
#     ----------------------------------------------------------
#     * imfs: intrinsic mode functions and remainder, shape = (order+1,x.shape).

#     References:
#     --------------------------------------------------
#     [1] N. E. Huang et al., "The empirical mode decomposition and the Hilbert 
#        spectrum for nonlinear and non-stationary time series analysis", 
#        Proc. R. Soc. Lond. A, Math. Phys. Sci., vol. 454, no. 1971, 903–995, (1998).
#     [2] N. E. Huang, "Hilbert-Huang transform and its applications", vol. 16. World Scientific, 2014.
#     [3] Z. Wu, N. E. Huang, "Ensemble empirical mode decomposition: 
#         A noise-assisted data analysis method", Adv. Adapt. Data Anal., vol. 1, no. 1, 1–41 (2008).
#     [4] J. Zheng, J. Cheng, Y. Yang, "Partly ensemble empirical mode decomposition: 
#         An improved noise-assisted method for eliminating mode mixing", 
#         Signal Process., vol. 96, 362–374, (2014).

#     '''
#     x   = np.asarray(x)
#     if order is None: order = x.shape[0]    
    
#     imf = np.zeros((order+1, x.shape[0]), dtype = x.dtype)
#     out = np.zeros(x.shape[0], dtype = x.dtype)
    
#     for i in np.arange(order): 
#         out = np.array(x - np.sum(imf,axis=0))
        
#         for _ in np.arange(max_itter):
#             envdw, envup, points = _envelops(out, method = method)

#             if stop_criteria(out, envdw, envup, points): break     
#             else: out -= 0.5*(envdw+envup)
        
#         imf[i,:] = out
#         (pmax,pmin,pzeros) = points
#         if(pmax.size < 2 or pmax.size < 2):
#             break
#     imf[i+1,:] = np.array(x - np.sum(imf,axis=0))
#     return imf

# #--------------------------------------------------------------------
# def _spline(values, points, new_points, method = 'cubic'):
    
#     if(method=='cubic'):
#         cofs = scipy.interpolate.splrep(points, values)
#         return scipy.interpolate.splev(new_points, cofs)
#     elif(method=='akim'):
#         return scipy.interpolate.Akima1DInterpolator(points,values)(new_points)
#     elif(method=='rbf'):
#         return scipy.interpolate.Rbf(points,values, function='gaussian')(new_points)
#     elif(method=='thin_plate'):
#         return scipy.interpolate.Rbf(points,values, function='thin_plate')(new_points) 
#     elif(method=='linear'):
#         return scipy.interpolate.Rbf(points,values, function='linear')(new_points)     
    
# #--------------------------------------------------------------------
# def _extension(values, points, n_points=2): 
#     '''
#     Mirror extention
#     '''
#     N = values.shape[0]
#     values = np.concatenate(( values[n_points-1::-1], values, values[N-1:N-n_points-1:-1] ))
#     points = np.concatenate(( 2*points[0] - points[n_points:0:-1], points, 2*points[-1] - points[N-2:N-n_points-2:-1] ))
#     return values, points

# #--------------------------------------------------------------------
# def _specialpoints(x,order=2, boundaires = False):
#     '''
#     Find special points (zeros, maximums and minimums) of the inpute
#     sequence.

#     Parameters:
#     ----------------------------------
#     * x: input sequence.
#     * order: number of points before and after point to determine the class.
#     * boundaires: if True, boun points (zero and last will also be concidered).
    
#     Returns:
#     -------------------------------
#     * pmax: point of maximums (peaks).
#     * pmin: point of minimums (peaks).
#     * pzero: point of zeros (minimums of |x|).
    
#     Notes:
#     ---------------------------------
#     * It is recommended to use _add_boundaries for bound points.
    
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]
    
#     if(order<1 or order>N//2):
#         raise ValueError('order should be between 1 and much less than samples size')

#     pmax  = np.array([], dtype = np.int)
#     pmin  = np.array([], dtype = np.int)
#     pzero = np.array([], dtype = np.int)
    
#     x_extend = np.concatenate((np.zeros(order), x, np.zeros(order) ))

#     #TODO: replace x on x_extend
#     for p in np.arange(order,N+order): #same as while(p<N+order)       
        
#         if(p-order>0 and p-order<N-1) or (boundaires):            
#             prange = x_extend[p-order:p+order]    
#             #max
#             if(x[p-order] == np.max(prange)):
#                 pmax = np.append(pmax, p-order)
#             #min
#             if(x[p-order] == np.min(prange)):
#                 pmin = np.append(pmin, p-order)
#             #zero
#             if(np.abs(np.real(x[p-order]))) == np.min(np.abs(np.real(prange))):
#                 pzero = np.append(pzero, p-order)            

#     return pmax, pmin, pzero

# #--------------------------------------------------------------------
# def _envelops(x, method = 'cubic'):

#     x = np.asarray(x)
#     N = x.shape[0]

#     pmax, pmin, pzeros = _specialpoints(x,order=2)
#     x_max, x_min = x[pmax],x[pmin]

#     envdw = np.zeros(N)
#     envup = np.zeros(N)
                        
#     if(np.min([pmax.size,  pmin.size])>=_MIN_EXTREMUMS):               
#         x_max,pmax = _extension(x_max,pmax)
#         x_min,pmin = _extension(x_min,pmin)

#         fp = np.min([pmax[0], pmin[0], 0])
#         lp = np.max([pmax[-1],pmin[-1],N])

#         n  = np.arange(fp,lp) 
#         envdw = _spline(x_min, pmin, n, method)[np.abs(fp):N+np.abs(fp)]
#         envup = _spline(x_max, pmax, n, method)[np.abs(fp):N+np.abs(fp)]

#     return envdw, envup, (pmax, pmin, pzeros)


# #--------------------------------------------------------------------
# def stop_criteria(out, envdw, envup, points):
#     ''' 
#     Cashy Criteria, monotonic criteria, IMF criteria.
#     '''
#     (pmax,pmin,pzeros) = points
#     if(pmax.size < 2 or pmax.size < 2):
#         return True  
    
#     if(abs(pmax.size + pmax.size - pzeros.size)<=1 ):
#         return True 
    
#     elif np.sum(0.25*np.square(envdw+envup))/np.sum(np.square(out))<TOL:
#         return True    
    
#     else:
#         return False