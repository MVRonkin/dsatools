import numpy as np
import scipy  

from ... import utilits as ut

__all__ = ['ewt']

def ewt(x, order = None, gamma=0, average=None ):
    '''
    Emperical Wavelet Transform (EWT).
    
    Parameters
    ------------------
    * x: 1d ndarray.
    * order: int or None,
        number of components to search,
        if order is None all possible imfs will be returned.
        FOR TEST: if order <1 than only peaks with value
        higher than order*max(sp) will be concidered.
    * gamma: float,
        filter parameter, than smaller it value,
        than higher each component bandwidth.
    * average: int or None,
        if not None it will be taken an additional 
        moving average with square window  
        for spetrum during its maximums search. 
    
    Return
    ------------------------
    * imfs: 
        intrinsic mode functions and remainder, 
        shape = (order+1,x.shape).
        if order is None all possible imfs will be returned. 

    Notes
    --------------------------
    * EWT is calculated as the following:
      * Determining the maximums of x(t) spectrum and the minimums 
        of x(t) spectrum between its maximums, which describes the 
        bandwidths of each spectrum component.
    
     * For each band the corresponding component is calculated 
       by convolution (conjugated multiplication 
        in the frequency domain) with the corresponding 
        Mayer wavelet (Littlewood wavelet 
        in the original paper), which is calculated as follows:
        ..math::
        
        wt_i(ω) = 1, 
            for ω∈[(1+γ) ω_n,(1-γ) ω_(n+1)]
        
        wt_i(ω) = cos⁡{[0.5πβ([(ω-ω_n (1-γ)])⁄(2γω_n ))]},
            for ω∈[(1-γ) ω_n,(1+γ) ω_n]
        
        wt_i(ω) = sin⁡{[0.5πβ(([ω-ω_(n+1) (1-γ)])⁄(2γω_(n+1) ))]}, 
            for ω∈[(1-γ)ω_(n+1),(1+γ)ω_(n+1)] 
                        
      where 
      * β(x)=x^4 (35-84x+70x^2-20x^3 ) for  x<1 and β(x)=1 for   x≥1; 
      * γ is the coefficient of filter bandwidth (wavelet bandwidth);
      * ω_n,ω_(n+1) are the start and end points of each bandwidth. 
    
      It has to be noted, that near-zero frequencies 
        were not considered in differ to the original paper.

    
    References
    --------------------------
    [1a] J. Gilles, "Empirical Wavelet Transform," 
        in IEEE Transactions on Signal Processing, 
         vol. 61, no. 16, pp.3999-4010, 2013, 
         doi: 10.1109/TSP.2013.2265222.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
            42141-empirical-wavelet-transforms 
    [2]  additionally see https://pypi.org/project/ewtpy/
    
    
    See also: 
    --------------------------
    vmd, 
    emd,
    hvd
    
    
    '''    
    x = np.asarray(x)
    N = x.shape[0]
    if(gamma is None): gamma = 1/N

    sp = np.fft.fft(x)
    
    #TODO:use Savatsky_Glory insted of this?
    if(average is not None): # alternative scipy.ndimage.filters.gaussian_filter(np.real(signal3),3.9130)
        average = int(abs(average))
        filter_ = np.fft.fft(np.ones(average)/average,N)
        sp_abs = np.abs(sp*np.conj(filter_))
    
    else:
        sp_abs = np.abs(sp)

    boundaries = boundaries_detect(np.abs(sp), 
                                   order, 
                                   minimums=True,
                                   sp_clear = sp_abs)
    
    #  filter_bank = np.zeros((boundaries.shape[0], N)) # additionaly filter bank can be saved.

    imfs = np.zeros((boundaries.shape[0]-1, N), dtype = x.dtype)
    
    for i,(low,high) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        
        if(low==N//2): break # if order higher than number of peaks.
 
        filter_ =  lp_wavelet(low,high,gamma,N)
        
        imfs[i,:] = np.fft.ifft(np.conjugate(filter_)*sp)
    
    return imfs 

#-----------------------------------------------------------------------------
def boundaries_detect(sp, order=None, minimums=True, sp_clear = None ):
    '''
    Function for search frequency bands for each 
        empirically determined signal.
    
    Parameters
    --------------------------------------
    * sp: 1d ndarray,
        is the input spectrum.
    * order: int or None,
        number of components to search.
    * minimums: bool,
        if true, than bands will 
        be determined by minimum points 
        between determined peaks.
    * sp_clear: int or None,
        if not None, then minimums will 
       be determined by minimums.
    
    Returns
    -----------------------------------------------
    * boundaries: 1d ndarray of int,
        points of start/stop each bandwidth. 
    
    References
    -----------------------------------------------
    [1a] J. Gilles, 
         "Empirical Wavelet Transform," 
         in IEEE Transactions on Signal Processing, 
         vol. 61, no. 16, pp. 3999-4010, Aug.15, 2013, 
         doi: 10.1109/TSP.2013.2265222.
    [1b] https://www.mathworks.com/matlabcentral/
        fileexchange/42141-empirical-wavelet-transforms 
    '''
    sp = np.asarray(sp)
    N = sp.shape[0]//2    
    peak_threshold = 0
    
    sp[:int(N//100+1)]=0 # Remove Global trend, can be eluminated
    
    # for search minimums of fillterd signal
    if(sp_clear is None): sp_clear = sp
    
    if order is not None:
        
        if abs(order)<1:
            peak_threshold = np.max(sp)*order
            order= None
        
        else:
            order= int(abs(order))
    
    #find by maximums
    #TODO: scipy.signals.arglmax in twice faster!
    # learn if scipy.signal.find_peaks also faster or not
    pp = ut.findpeaks(sp,
                      order=2, 
                      mode = 'maximum', 
                      min_distance = 1, 
                      peak_threshold=peak_threshold)#scipy.signal.argrelmax(sp)[0]

    # chooes order value number of max intensity components.
    if order is not None:
        order = int(order)
        idxs  = np.argsort(sp[pp])[::-1]   
        pp    = np.sort(pp[idxs][:min(order,pp.size)])
        order -=1
    
    # preliminary determining minimums 
    # as middle point between maximums
    pp = (pp + np.concatenate(([0],pp[:-1])))//2
    
    # add zero and last points to set bounds of components.
    pp = np.concatenate([[0],pp,[N-1]])

    # we do not concidering local minimums or maxmin due 
    # to not trust to points in noises     
    # search the smallest minimums near maximums as boundaries
    if (minimums):
        for i in range(1,pp.size-1):
            neighb_low = int(pp[i]-(np.abs(pp[i]-pp[i-1])/2))
            neighb_hi  = int(pp[i]+(np.abs(pp[i+1]-pp[i])/2))

            imini   = np.argmin(sp_clear[neighb_low:neighb_hi])
            pp[i-1] = imini+neighb_low;
        pp = pp[:-1]#last point N-1

    pp = np.sort(pp) #pp[-2],pp[-1] = pp[-1],pp[-2]!
    
    #add to required order
    if order is not None and (order > pp.size):
        pp = np.append(pp, N*np.ones(order-pp.size))
    
    return np.array(pp,dtype=np.int) 

#-----------------------------------------------------------------------------
def _beta(xs):
    '''
    Reference function for Mayer wavelet.
    
    References:
    -----------------------------------------------
    [1a] J. Gilles, "Empirical Wavelet Transform," 
         in IEEE Transactions on Signal Processing, 
         vol. 61, no. 16, pp. 3999-4010, Aug.15, 2013, 
         doi: 10.1109/TSP.2013.2265222.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
                                42141-empirical-wavelet-transforms 
    [2]  https://en.wikipedia.org/wiki/Meyer_wavelet
    '''
    bm = np.ones(xs.size)
    idxs = np.flatnonzero(xs<1)
    x= xs[idxs]
    bm[idxs]=(x**4)*(35.-84.*x+70.*(x**2)-20.*(x**3))
    return bm

#-----------------------------------------------------------------------------
def lp_wavelet(low,high,gamma,n_sig):
    '''    
    Envelope part of mayer Wavelet (LittleWood-Paley wavelet).
    
    References:
    ------------------------------------------------
    [1a] J. Gilles, 
         "Empirical Wavelet Transform," 
         in IEEE Transactions on Signal Processing, 
         vol. 61, no. 16, pp. 3999-4010, Aug.15, 2013, 
         doi: 10.1109/TSP.2013.2265222.
    [1b] https://www.mathworks.com/matlabcentral/fileexchange/
                            42141-empirical-wavelet-transforms 
    [2]  https://en.wikipedia.org/wiki/Meyer_wavelet
    '''
    out=np.zeros(n_sig)    
    #high_pass part
    cof_high = 1./(2*gamma*high)    
    hp_lp=int((1.+gamma)*low)
    hp_fp=int((1.-gamma)*low)
    #low_pass part
    cof_low  = 1./(2*gamma*low)
    lp_lp=int((1.+gamma)*high)
    lp_fp=int((1.-gamma)*high)

    if(lp_fp>=hp_lp): out[hp_lp:lp_fp] = 1 
    if(hp_fp<=hp_lp): out[hp_fp:hp_lp] = \
        np.sin(np.pi*_beta(cof_low*(np.arange(hp_lp-hp_fp)))/2)
    if(lp_fp<=lp_lp): out[lp_fp:lp_lp] = \
        np.cos(np.pi*_beta(cof_high*(np.arange(lp_lp-lp_fp)))/2)

    return out