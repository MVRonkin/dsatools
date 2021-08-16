import numpy as np
import scipy

#-------------------------------------------------------------------
def findpeaks(x,order=1, mode = 'maximum', min_distance = None, peak_threshold=0):
    '''
    Find peaks or zeros of the input samples.
    
    Parameters
    -------------
    * x: 1d ndarray.
    * order: int,
        number of samples before and after 
        point to classify it as peak.
    * mode: string,
        mode of search,
        mode = {'maximum' 'minimum','zeros','maxabs'}; 
        maxabs=max by module.
    * min_distance: int or None, 
        minimal distance between peaks in points.
        if min_distance is None, min_distance = order
    * peak_threshold: float,
        threshold value of samples intencity (in absolute values).
    
    Returns
    -----------
    * positions:1d ndarray,
        positions of the peaks.  
    
    Notes
    ----------
    * Mindistance have to be at least equal to order.
    
    '''    
    x = np.asarray(x)
    N = x.shape[0]
    
    if(order<1 or order>N//2):
        raise ValueError('order should be between 1 and much less than samples size')
    
    if min_distance is None: min_distance = order
    elif(min_distance < order): min_distance = order
        
    if (mode == 'maximum'):
        fn = lambda xi, x:xi==np.max(x)
    
    elif(mode == 'minimum'):
        fn = lambda xi,x:xi==np.min(x)

    elif(mode == 'zeros'):
        fn = lambda xi,x:np.abs(np.real(xi)) == np.min(np.abs(np.real(x)))
    
    elif(mode == 'maxabs'):
        fn = lambda xi,x:np.abs(np.real(xi)) == np.max(np.abs(np.real(x)))    
    
    positions = np.array([], dtype = np.int)
    
    x_extend = np.concatenate((np.zeros(order), x, np.zeros(order)))
#     x_extend[x_extend<peak_threshold] = peak_threshold
#     plt.plot(x_extend)
    
    p=order
    for _ in np.arange(N): #same as while(p<N+order)       
        prange = x_extend[p-order:p+order]        
        
        if( fn(x[p-order],prange) ):
            if((x[p-order]) >peak_threshold):
                positions  = np.append(positions, p-order)
                p += min_distance
        
        p+=1
        if(p>=N+order):break
    
    #TODO: check conditions at the start and at the end.
    
    return positions