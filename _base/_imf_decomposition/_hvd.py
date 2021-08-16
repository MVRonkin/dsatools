import numpy as np
import scipy  

from ... import operators

def hvd(x,order, fpar=20, ret_freqs = False ):
    '''
    Hilbert Vbration Decomposition (HVD).
    
    Parameters
    -------------------------------
    * x: 1d ndarray.
    * order: int,
        number of component to decompose.
    * fpar: int,
        filter parameter, equal to 
        point of cut frequency for low-pass filter
        (have to be regulized for optimal decomposition).
    * ret_freqs: bool,
        return centrel frequencies for each component.
    
    Return
    ------------------------------------
    > if ret_freq = False , 
    * components:2d ndarray,
             calculated imfs.
    > else
    * freqs: 1d ndarray,
           frequencies.
    
    Notes
    -----------------------
    * Function in TEST, now it can work only with
      narrowband signals!
    * Decomposition results are very depends on the
        fpar value.
    
    References
    ------------------------
    [1a] M. Feldman, "Time-Varying Vibration Decomposition 
         and Analysis Based on the Hilbert Transform". 
         Journal of Sound and Vibration. 
         2006, Vol 295/3-5 pp. 518-530.         
    [1b] https://ht.net.technion.ac.il/matlab-simulink/
    [1c] http://hitech.technion.ac.il/feldman/hvd.html
    [2] J.J. Ramos, J.I. Reyes E. Barocio, 
        "An improved Hilbert Vibration Decomposition method for  
        analysis of low frequency oscillations", 2014 IEEE PES 
        Transmission & Distribution Conference and Exposition
        - Latin America (PES T&D-LA), Medellin, 2014, 
        pp. 1-6, doi: 10.1109/TDC-LA.2014.6955216.
    
    See also
    --------------------
    * emd
    * vmd
    * ewt
    
    '''
    
    x = np.asarray(x)

    N = x.shape[0]
    
    n = np.arange(N)
    
    fpar = int(fpar)
    
    #TODO: alternatively use band filtration of the second harmonic
    # can be more stable
    hp = square_window(N,wfilt=[0,fpar], real_valued_filter=True)

    f = np.zeros(order)
    imf = np.zeros((order,N),dtype = x.dtype)
    
    x_rest = np.array(x)#copy
    
    for i in range(order):

        phase = operators.arg(x_rest)
        intfreq = operators.diff(phase)/2/np.pi
        intfreqf = filter_by_window(intfreq,hp)
        
        # to avoid transition zone
        f[i] = np.abs(np.mean(intfreqf[N//50:-N//50]))

        x_ref = np.exp(-2j*np.pi*f[i]*n)

        x1 = filter_by_window(x_rest*x_ref,hp)

        env = np.abs(x1)

        phase0 = np.angle(x1)

        imf[i,:] = env*np.exp(1j*(2*np.pi*f[i]*n+phase0))

        x_rest -= imf[i,:]
        
    if ret_freqs:
        return f
    else:
        return imf
    
    
#-------------------------------------------------------------
def square_window(N,wfilt, real_valued_filter=True):
    '''
    square window in range 0:fs//2.
    
    Parameters
    -------------
    * N: int,
        filter length (sample length).
    * wfilt: [int,int],
        cut-off low and high points.
    
    Returns
    --------
    * H: window
    
    '''  
    lp,fp = wfilt
    
    if lp>N:lp = int(N)                  
    if lp-fp<0:lp,fp=fp,lp


    One   = np.ones(lp-fp)             
    z2    = np.zeros(N-lp)             
    
    if fp==0:  Hp = np.hstack((One, z2))    
                                        
    else:
        z1 = np.zeros(fp);              
        Hp = np.hstack((z1,One, z2))
    
    if real_valued_filter:    
        Hp = make_window_real_valued(Hp, N)    
    
    return   Hp
#-------------------------------------------------------------
def make_window_real_valued(H, N):
    '''
    Make window real valued.
    
    Parameters
    -------------
    * H: 1d ndarray,
        spectrum window (in frequency domain).
    * N: int,
        filter length (sample length).

    Returns
    --------
    * H: 1d ndarray,
        spectrum window (in frequency domain)
        with two part in range 0:fs//2 and fs//2:fs 
        (mirrowed and shifted on 1 point to fs//2).
    '''
    H[N//2+1:N] = H[1:N//2][::-1]    
#     Hp[length//2-1:length//2+1]=Hp[length//2-1:length//2+1]*1/2    #or in other versions 0
    return H
#-------------------------------------------------------------
def filter_by_window(s_in,H):
    '''
    Filter by spectrum window.
    
    Parameters
    -------------
    * s_in: 1d ndarray,
        input signal (in time domain).        
    * H: 1d ndarray,
        spectrum window (in frequency domain).
    
    Returns
    ------------
    * s_out: 1d ndarray,
        filtered signal (in time domain).
    
    '''
    s_in   = np.asarray(s_in)
    N      = int(s_in.shape[0])
    
    Sp     = np.fft.fft(s_in)     
    Sp     = Sp * np.conj(H[0:N])  
    s_out  = np.fft.ifft(Sp)    
    
    return   s_out