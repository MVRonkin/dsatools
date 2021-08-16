import numpy as np
import scipy.signal

from ... import operators

__all__ = ['FitzR','Tretter_f','Kay','MCRB','MandM','Fitz','FandK','maxcorfreq_real']
#--------------------------------------------------------------
def _check_input(s,fs=None):
    s = np.array(s)
    N = s.shape[0]
    
    if s.dtype not in \
        [complex,np.complex,np.complex64,
		np.complex128,np.complex_]:
        s = scipy.signal.hilbert(s)
        
    if(fs is None):
        fs = N
        
    return s,N,fs
#--------------------------------------------------------------
def maxcorfreq_real(s,fs=None,with_xcorr=True): 
    '''
    Fast signal frequency estimater for real-valued
      single-tone short sample-size signals on the 
      background of white gaussian noises with high 
      signal-to-noise ratio (SNR).

    Parameters
    -----------
    * s: 1d ndarray (float),
        is the input signal.
    * fs: float or int,
        is the sampling frequency.

    Returns
    --------------------
    * f: float,
        estimated frequency.

    Notes
    -----------
    * Than higher sample size (number of zero-crossing),
        than lower SNR is required.
    * Algorithm is based on properties of correlation
        coefficient.
    * For estimate frequrncy in
        points multupy result on 2*pi

    Example
    -------------
    import numpy as np
    import matplotlib.pyplot as plt
    import dsatools
    import dsatools.utilits as ut
    
    delay=5
    f0=1
    delta_f=0.5
    fs = 10
    SNR = 50
    signal =\
        dsatools.generator.beatsignal(delay=delay,
                                      f0=f0,
                                      delta_f=delta_f,
                                      fs=fs,
                                      snr_db=SNR).real
    ut.probe(signal)
    Tm = len(signal)/fs
    f_exp = delay*delta_f/Tm
    print(maxcorfreq_real(signal, fs=fs),f_exp)
    '''
    
    s = np.array(s)
    if fs is None:
        fs = s.size
    
    if with_xcorr:        
        s  = operators.correlation(s,mode='full').real
        
    s1 = s[1:]
    s2 = s[:-1]

    corcof = np.sum(s1*s2)/np.sum(np.square(s))
    angle = np.arccos(corcof) 
    return fs*angle/(2*np.pi)

#--------------------------------------------------------------
def maxcorfreq(s,fs=None, with_xcorr=True): 
    '''
    Fast signal frequency estimater for complex-valued
      single-tone short sample-size signals on the 
      background of white gaussian noises with high 
      signal-to-noise ratio (SNR).

    Parameters
    -----------
    * s: 1d ndarray (float),
        is the input signal.
    * fs: float or int,
        is the sampling frequency.
    * with_xcorr: bool,
        if true frequency of correlation function
        will be estimated.
        
    Returns
    --------------------
    * f: float,
        estimated frequency.

    Notes
    -----------
    * Than higher sample size (number of zero-crossing),
        than lower SNR is required.
    * Algorithm is based on properties of correlation
        coefficient.
    * For estimate frequrncy in
        points multupy result on 2*pi.
        
    Example
    -------------
    import numpy as np
    import matplotlib.pyplot as plt
    import dsatools
    import dsatools.utilits as ut
    
    delay=5
    f0=1
    delta_f=0.5
    fs = 10
    SNR = 50
    signal =\
        dsatools.generator.beatsignal(delay=delay,
                                      f0=f0,
                                      delta_f=delta_f,
                                      fs=fs,
                                      snr_db=SNR)
    ut.probe(signal)
    Tm = len(signal)/fs
    f_exp = delay*delta_f/Tm
    print(maxcorfreq(signal, fs=fs),f_exp)
    '''
    s,N,fs = _check_input(s,fs)
    
    if with_xcorr:        
        s  = operators.correlation(s,mode='straight')
        
    s1 = s[1:]
    s2 = s[:-1]
    
    corcof = np.sum(s1*np.conj(s2))#/np.sum(np.square(s))
    angle = np.angle(corcof) 
    return fs*angle/(2*np.pi) 
 

#--------------------------------------------------------------  
def fitz_r(s,fs = None, w_on = True):
    '''
    Frequency estimator based on the phase-to-time approximation
      of signal correlation function by weighted least-square with 
      correlation of signal modules as weights function.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    * w_on: bool,
        If False, than |R| will be changed on ones array,
        (need for tests of unгтiform |R| hypothesis).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.
    
    Notes
    -----------------------
    * If signal real-valued use hilbert transfrom.
    * If fs = N, then f will be measured in points.
    * Estimator is based on the  
        ..math::
        f = (fs/2\\pi) \\sum_n{n|R(n)|arg(R(n))}/sum_n{n^2|R(n)|)}
        
        where:
        * f is the estimated frequency (in Hz).
        * R(n) is the straight part of the 
                autocorrelation function (biased).
        * arg(R(n)) = unwrap (angle R(n)).
        * n = 0,...,N-1, (N is the input length).
        * fs is the sampling frequency.
    
    Example
    -------------------

    
    References
    -------------------
    [1] Ronkin M.V., Kalmykov A.A., 
        "Phase based frequency estimator for short range F
        MCW radar systems",
        2018 Ural Symposium on Biomedical Engineering, 
        Radioelectronics and Information Technology (USBEREIT), 
        2018. pp. 367-370.
        https://ieeexplore.ieee.org/document/8384625
        DOI: 10.1109/USBEREIT.2018.8384625. 
    [2] Fu Р., Kam P.Y., 
        "Sample-autocorrelation-function-based frequency estimation 
        of a single sinusoid in AWGN", 
        IEEE 75th VTC Spring, 2012.
        https://ieeexplore.ieee.org/document/6239864
        DOI: 10.1109/VETECS.2012.6239864.
    [3] Fitz M.P., 
        "Further results in the fast estimation of a single frequency",
        IEEE trans. on communications.
        1994. v 42. p. 862-864.
    
    '''
    s,N,fs = _check_input(s,fs)
 
    R      = operators.correlation(s,mode='straight')
    n      = np.arange(1,N)
    n2     = n**2
    
    dR     = R[1:]*np.conj(R[0:N-1])
    incR   = np.angle(dR) #np.arctan2(np.imag(dR),np.real(dR))     
    
    if w_on:        
        Wdenum = np.sum(np.square(n)*np.abs(R[1:]))
        Wnum   = n*np.abs(R[1:])
        fres   = np.sum(Wnum*np.cumsum(incR)/Wdenum) #unwrapAngleR = np.cumsum(incR)
    
    else:        
        fres1 = np.sum(incR)/(2*N-1)
        fres2 = np.sum(n*(n-1)*incR)/((2*N-1)*N*(N-1))
        fres  = 3*(fres1-fres2)

    fres   = fs*fres/2/np.pi
    return   fres

#-------------------------------------------------------------- 
def tretter_f(s,fs = None):
    '''
    FUNCTION FOR TESTS.
    Freuquency estimation base on Tretter signal model.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.
        
    Notes
    -------
    * If signal real-valued use hilbert transfrom.
    * If fs = N, then f will be measured in points.
    * Function FOR TEST ONLY - closed form has been derived 
        with some simplifications which lead to some bias
        in estimation.
    * The estimator:      
      ..math::
      f = (fs/2\\pi)6\\sum{(N-2n)arg(s(n))}/N^3
    
      where:
      * f is the estimated frequency (in Hz);         
      * arg(s(n)) = unwrap(angle s(n));
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency.    
    
    References
    ---------------
    [1] Tretter S. A., 
        Estimating the frequency of a noisy sinusoid by linear regression, 
        IEEE Trans. Inform. Theory..
        1985. v. IT-3 1. p. 832-835.
    
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
        
    '''
    s,N,fs = _check_input(s,fs)    
    n      = np.arange(N)
    w      = (N-2*n)/N**3
    fres   = 6*np.sum(w*np.unwrap(np.angle(s)))
    fres   = -fs*fres/2/np.pi
    return   fres    

#--------------------------------------------------------------
def kay(s,fs=None):    
    '''
    Kay S. Central finite-difference 
    (CFD) frequency estimator.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.  
    
    Notes
    -----------------------------------
    * If signal real-valued use hilbert transfrom.
    * if fs = N, then f will be measured in points.
    * Estimator have the following expression:    
    * The frequency estiamtor.
      ..math::
      f = (fs/2\\pi)\\sum(w(n)*s[n]*conj(s[n-1])),
      where:
      * f is the estimated frequency (in Hz);         
      * w(n) = (1.5N/(N^2-1))(1 - ((n-0.5N+1)/(0.5N))^2);
      * arg(s(n)) = (angle s(n));
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency.

    References
    ------------------------
    [1] Kay S.,
        A Fast and Accurate Single Frequency Estimator, 
        IEEE transactions on acoustics. Speech. And signal processing. 
        1989. v. 37. № 12. p. 1987-1990.
    [2] Kay, S., 
        Comments on “Frequency estimation by linear prediction,” 
        with Authors: Reply by L. Jackson and D. Tufts,
        IEEE Trans. Acoustics Speech Signal Process. 
        1989. v.27. p. 198 –200.
    [3] Fowler M.L.,
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
        
    '''
    s,N,fs = _check_input(s,fs)    
    n      = np.arange(N)
    w      = (1.5*N/((N**2)-1))*(1-((n-0.5*N+1)/(N/2))**2)    
    s_prod = s[1:]*np.conj(s[0:N-1])        
    fres   = np.sum(w[0:N-1]*np.angle(s_prod))
    fres   = fs*fres/2/np.pi
    return   fres

#--------------------------------------------------------------
def fitz(s,fs=None):
    '''
    Frequency estimator based on the M. Fitz estimator.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.  
    
    Notes
    -----------------------------------
    * If signal real-valued use hilbert transfrom.
    * if fs = N, then f will be measured in points.
    * Estimator have the following expression:        
      ..math::
      f = (fs/2\\pi) \\sum_n{n*arg(R(n))}/sum_n{n^2)}
      where:
      * f is the estimated frequency (in Hz);
      * R(n) is the straight part of the autocorrelation function (biased);
      * arg(R(n)) = unwrap (angle R(n));
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency. 
    
    References
    ----------------------
    [1] Fitz M.P., 
        Further results in the fast estimation of a single frequency,
        IEEE trans. on communications.
        1994. v 42. p. 862-864.
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
    
    '''
    s,N,fs = _check_input(s,fs) 
    R      = operators.correlation(s,mode='straight')
    n      = np.arange(N)
    n2     = n**2
    Wnum   = np.sum(n*np.unwrap(np.angle(R)))
    Wdenum = np.sum(n2)
    fres   = Wnum/Wdenum
    fres   = fs*fres/2/np.pi
    return   fres

#--------------------------------------------------------------
def m_and_m(s,fs=None):
    '''
    Frequency estimator based on 
            the Mengali U., Morelli M., estimator.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.  
    
    Notes
    -----------------------------------
    * If signal real-valued use hilbert transfrom.
    * if fs = N, then f will be measured in points.
    * Estimator has the following expression:
      ..math::
      f = (fs/2\\pi) \\sum_n{w(n)*arg(R'(n))}
      where:
      * f is the estimated frequency (in Hz);
      * w(n) = 3(N-n)(N-n+1)/(N(N^2-1));
      * R'(n) = R[n]*np.conj(R[n-1]);
      * R(n) is the straight part of the autocorrelation function (biased);
      * arg(R'(n)) = angle R'(n);
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency
    
    References
    --------------------------------
    [1] Mengali U., Morelli M., 
        Data-aided frequency estimation for burst digital transmission, 
        IEEE Trans. Commun.
        1997. v. 45, 1997. p. 23-25.
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
    '''    
    
    s,N,fs = _check_input(s,fs) 
    R      = operators.correlation(s,mode='straight')
    n      = np.arange(N)
    Wnum   = 3*(N-n)*(N-n+1)
    Wdenum = N*(N**2-1)
    dR     = R[1:]*np.conj(R[0:N-1])
    fres   = np.sum(Wnum[1:]*np.angle(dR))/Wdenum
    fres   = fs*fres/2/np.pi
    return   fres

#--------------------------------------------------------------
def mcrb(s,fs=None):
    '''
    Frequency estimator based on the 
            Dongming B., Gengxin Z., Xinying Y. estimator.
     
    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.  
    
    Notes
    -----------------------------------
    * If signal real-valued use hilbert transfrom.
    * if fs = N, then f will be measured in points.
    * Estimator has the following expression:
      ..math::    
      f = (fs/2\\pi) \\sum_n{w(n)*arg(R'(n))} 
      where:
      * f is the estimated frequency (in Hz);
      * w(n) = [N(N^2-1)-n(n-1)(3N-2n+1)]/(N(N^2-1));
      * R'(n) = R[n]*np.conj(R[n-1]);
      * R(n) is the straight part of the autocorrelation function (biased);
      * arg(R'(n)) = angle R'(n);
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency. 

    References
    -------------------------------------
    [1] Dongming B., Gengxin Z., Xinying Y.,
        A maximum likelihood based carrier frequency estimation algorithm,
        Proceedings of ICSP2000. 
        2000. v.1. p. 185 - 188.
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
    
    '''   
    s,N,fs = _check_input(s,fs) 
    R      = operators.correlation(s,mode='straight')
    n      = np.arange(N)
    Wnum   = N*(N**2-1)-n*(n-1)*(3*N-2*n+1)    
    Wdenum =  (N**2)*(N**2-1)
    dR     = R[1:]*np.conj(R[0:N-1])
    fres   = 2*np.sum(Wnum[1:]*np.angle(dR))/Wdenum   
    fres   = fs*(fres)/2/np.pi
    return   fres

#--------------------------------------------------------------
def f_and_k(s,fs=None):
    '''
    Fu Н., Kam P.Y. frequency estimator.

    Parameters
    --------------
    * s: 1d ndarray (complex),
        the input  signal.
    * fs: float or None,
        ampling frequency (fs = s.size, if None).
    
    Returns
    --------------------
    * f: float,
        estimated frequency.  
    
    Notes
    -----------------------------------
    * If signal real-valued use hilbert transfrom.
    * if fs = N, then f will be measured in points.
    * Estimator has the following expression:
    ..math::
    f = [1/Psi]*
        [sum(|s|)*sum (n|s|arg(s))-sum(n|s|)sum(|s|arg(s))],
      where:        
      * f is the estimated frequency (in Hz);         
      * Psi = sum(|s|)*sum(n^2|s|)-[sum(n|s|)]^2;
      * arg(s(n)) = unwrap(angle s(n));
      * n = 0,...,N-1, (N is the input length);
      * fs is the sampling frequency.
    
    Refernces
    --------------------------
    [1] Fu Н., Kam P.Y. 
        Linear estimation of the frequency and phase of a noisy sinusoid,
        IEEE Trans. on Inform. Theory. 
        2003. V.31. p. 832 - 835.
    [2] Fu Р. Kam P.Y.,
        ML Estimation of the Frequency and Phase in Noise, 
        Proceeding of IEEE Globecom 2006. 
        2006. p. 1-5.   
        
    '''    
    s,N,fs = _check_input(s,fs) 
    n      = np.arange(N)
    ms     = np.abs(s)
    alpha  = np.sum(ms)    
    beta   = np.sum(n*ms)
    n2     = n**2
    etta   = np.sum((n2)*ms)
    Psi    = alpha*etta-beta**2
    args   = np.unwrap(np.angle(s))
    fres   = alpha*np.sum(n*ms*args)-beta*np.sum(ms*args)
    fres   = fres/Psi
    fres   = fs*fres/2/np.pi
    return   fres

#-------------------------------------------------------------- 
def mlfft(s,fs = None,Nfft = None):
    '''
    Maximum-likelihood estimator of frequency,
      based on the maximum searching of the signal spectrum,
      obtained by fast Fourier transform with zero-padding.

    Parameters
    -----------------
    * s: is the input 1d ndarray (input signal);
    * fs: ampling frequency (fs = Nfft, if None);
    * Nfft: length of signal spectrum (with zerro-padding), 
                                            Nfft = s.size if None.

    
    Returns
    ------------------
    * f - estimated frequency.  
        
    Notes
    ---------------
    * if fs = N, then f will be measured in points.
    
    Referenes
    ------------------
    [1] Rife D. and Boorstyn R., 
        Single-tone parameter estimation from 
        discrete-time observations, 
        IEEE Transactions on Information Theory, 
        vol. 20, № 5, 1974, p. 591–598.    
    [2] Fowler M.L., 
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
        
    '''
    s,N,fs = _check_input(s,fs)
    
    if(not Nfft): Nfft = N
    if(not fs):   fs = Nfft        
        
    LimOfFind = int(Nfft/2)
    
    S      = np.abs(np.fft.fft(s,Nfft))  # расчет квадрата амплитуды спектра
    
    S      = np.hstack((np.zeros(1),S[1:LimOfFind]))
    
    pp     = np.flatnonzero(S==max(S))[0]   # поиск первого максимума 
    
    f_res  = fs*(pp)/Nfft      # расчет частоты первого максимума  
    
    return   f_res

#-------------------------------------------------------------- 
def cfd_est(s,fs=None,mode='FitzR'):
    '''
    United central-finite-difference (CFD) 
        frequency estimator function, based on the 
        phase-to-time approximation by least-square method.

    Parameters
    --------------
    * s: is the input 1d ndarray (input signal);
    * fs: ampling frequency (fs = s.size, if None);        
    * modes = ['FitzR','Fitz','Kay','FandK','M&M','MCRB',
                 'FullTest','FullTestLight','FullTestHard'],
      * if FullTest zero_padding in mlfft up to 2^18
      * if FullTestLight zero_padding in mlfft up to 2^10
      * if FullTestHard  zero_padding in mlfft up to 2^20.
    
    Notes
    -------------
    * If signal real-valued use hilbert transfrom.
    * Function include methods:
        'FitzR','Fitz','Kay','FandK','M&M','MCRB','mlfft'.
        the mlfft estimater perfermed by n_fft 
                                points with zerro_padding,
        see 'FullTest','FullTestLight','FullTestHard'.
      * if fs = N, then f will be measured in points. 
    
    Returns
    --------------
    * f - estimated frequency.
    
    References
    ---------------
    [1] Fowler M.L.,
        Phase-based frequency estimation: a review, 
        Digital Signal Processing. 
        2012. v. 12. p. 590–615.
    
    '''
    if (mode=='FitzR'):
        fest = FitzR(s,fs)
    elif (mode=='Fitz'):
        fest = Fitz(s, fs)
    elif (mode=='Kay'):
         fest = Kay(s, fs)
    elif (mode=='FandK'):
         fest = FandK(s, fs)
    elif (mode=='M&M'):
         fest = MandM(s,fs)
    elif (mode=='MCRB'):
         fest = MCRB_est(s,fs)
    elif (mode=='FullTest'):
         fest = _cfd_fulltest(s,fs,2**18)
    elif (mode=='FullTestLight'):
        fest = _cfd_fulltest(s,fs,2**10)
    elif (mode=='FullTestHard'):
         fest = _cfd_fulltest(s,fs,2**20)
    else:
         fest = FitzR_est(s,fs)
         print ('It is wrong type of estimation')       
    return    fest
#-------------------------------------------------------------- 
def _cfd_fulltest(s,fs,Nfft):
       fest    = np.zeros(7) 
       fest[0] = mlfft(s,fs,Nfft) 
       fest[1] = fitz_r(s,fs)
       fest[2] = fitz (s,fs)
       fest[3] = kay  (s,fs)
       fest[4] = f_and_k(s,fs)
       fest[5] = m_and_m(s,fs)
       fest[6] = mcrb (s,fs)
       return    fest



#     ############# CROSS-SCFD EST #############
# def BLUE_est(s1,s2,fs):
#     N      = len(s1)
#     n      = np.arange(N)
#     n2     = n**2
#     s      = np.conj(s1)*s2
#     R      = MyCorr(s,N)    
#     W      = np.abs(R)
#     Wnum   = n*W
#     Wdenum = np.sum((n2)*W)
#     fres   = np.sum(Wnum*arg(R))/Wdenum
#     fres   = fres*fs/2/np.pi
#     return   fres
    
# def BLUE0_est(s1,s2,fs):
#     N      = len(s1)
#     n      = np.arange(N)
#     n2     = n**2
#     s      = np.conj(s1)*s2
#     R      = MyCorr(s,N)
#     Wnum   = n
#     Wdenum = np.sum(n2)
#     fres   = np.sum(Wnum*arg(R))/Wdenum
#     fres   = fres*fs/2/np.pi
#     return   fres
