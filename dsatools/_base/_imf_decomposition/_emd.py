import numpy as np
import scipy  
import scipy.signal
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
       vol. 454, no. 1971, 903�995, (1998).
    [2] N. E. Huang, 
        "Hilbert-Huang transform and its applications", 
        vol. 16. World Scientific, 2014.
    [3] Z. Wu, N. E. Huang, 
        "Ensemble empirical mode decomposition: 
        A noise-assisted data analysis method", 
        Adv. Adapt. Data Anal., vol. 1, no. 1, 1�41 (2008).
    [4] J. Zheng, J. Cheng, Y. Yang, 
        "Partly ensemble empirical mode decomposition: 
        An improved noise-assisted method for eliminating mode mixing", 
        Signal Process., vol. 96, 362�374, (2014).

    
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
   
   
    for ord_cnt in range(order):
        h = x

        for cnt in range(max_itter):
            
            s1 =  get_envelope(h,  method = method)
            s2 = -get_envelope(-h, method = method)
            
            mean_env = (s1+s2)/2
            
            # for RBF interpolation envelope is complex
            if (x.dtype is complex) and (mean_env.dtype != complex):
                
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
        cofs = scipy.interpolate.splrep(points, values.real)
        return scipy.interpolate.splev(new_points, cofs)
    elif(method=='akim'):
        return scipy.interpolate.Akima1DInterpolator(points,values.real)(new_points)
    elif(method=='rbf'):
        return scipy.interpolate.Rbf(points,values.real, function='gaussian')(new_points)
    elif(method=='thin_plate'):
        return scipy.interpolate.Rbf(points,values.real, function='thin_plate')(new_points) 
    elif(method=='linear'):
        return scipy.interpolate.Rbf(points,values.real, function='linear')(new_points)   
#--------------------------------------------------------
def findpeaks(x):
    ''' find maximums of signals.
    '''
#     return scipy.signal.argrelmax(np.real(x))[0]
# def peaks(X):
    dX = np.sign(np.diff(x.transpose())).transpose()
    locs_max = np.where(np.logical_and(dX[:-1] > 0, dX[1:] < 0))[0] + 1

    return locs_max
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

