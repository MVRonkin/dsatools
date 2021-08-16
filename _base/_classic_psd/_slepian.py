import numpy as np
import scipy

__all__ = ['slepian_filter','slepian_psd']

def slepian_filter(length, filter_band_length, order=None):    
    ''' 
    Slepian filters bank.
    
    Parameters
    --------------------
    * length: int,
        the length pf signal.
    * filter_band_length: [int, int],
        filter bandwidth with respect to length 
        (in points).
    * order: int,
        order of the filter.
    
    Returns
    ------------------
    * Matrix of filter windows: 2d ndarray,
        base_band_ratio x order with Slepian 
        sequence in the colunms.
    
    Refernce
    ---------------
    [1a] P. Stoica, R.L. Moses, 
        Spectral analysis of signals 
        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
            - Dr.Moses Spectral Analysis of Signals: Resource Page.
    
    Examples
    ---------------
        h = slepian_filter(256,4,4)
        plt.plot(h[:,1])
        
    '''
    N = length
    K = filter_band_length
    
    if(order is None): order = filter_band_length
        
    vect = K/N*np.sinc(K/N*np.arange(N)) 
    gamma = scipy.linalg.toeplitz(vect)
    D,V=np.linalg.eig(gamma)

    h=V[:,:order]
    
    if np.sum(h[:,1])<0:
        h[:,1]=-h[:,1]
    
    return h
    
#---------------------------------------    
def slepian_psd(x, order, n_psd=None):
    '''
    Estimation of the pseudo-spectrum based on the 
        Slepian - Refil algorithm.
      
    Parameters
    -------------------
    * x: 1d ndarray.
    * order: int, 
        order of the model.
    * n_psd: int or None,
        number of samples in psd.
    
    Returns
    ------------------------
    * pseudo-spectrum: 1d ndarray.
      
    Refernce
   ------------------------
    [1a] P. Stoica, R.L. Moses, 
        Spectral analysis of signals 
        - New-York: Present-Hall, 2005.
    [1b] http://www2.ece.ohio-state.edu/~randy/SAtext/ 
         Dr.Moses Spectral Analysis of Signals: Resource Page.
    
    Examples
    ------------------------
    
    See also
    ------------------------
    capone
    
    
    '''
    x=np.array(x) 
    N=x.shape[0]  
    
    if(n_psd is None):
        n_psd=N
    
    h=slepian_filter(N,order,order)
    
    psd = np.zeros(n_psd)
    
    for i in np.arange(order):
        sp   = np.fft.fft(x*h[:,i],n_psd)
        psd += np.real(sp*np.conj(sp))# 0 imagenary part remain

    return psd

