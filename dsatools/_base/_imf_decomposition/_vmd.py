import numpy as np
import scipy  

__all__ = ['vmd']

def  vmd(x, 
         order, 
         reularization=2000, 
         noise_impact=0, 
         max_iter = 500, 
         tolerance=1e-7, 
         ret_freqs = False):
    '''
    Variational Mode Decomposition (VMD).    
    The decomposition of the signal on the 
       modes with the requirement of the most compact 
       spectrum bandwidth around 
       the central frequency of components.

    Parameters
    ----------------
    * x: 1d ndarray.
    * order: int,
        order of the model.
    * reularization: float,
        regularization parameter for 
        itterative optimization problem.
    * noise_impact: float,
        noise imact into optimization.
    * max_iter: int,
        maximum number of itterations.
    * tolerance: float, 
        Stop criteria tolerance.
    * ret_freqs: bool,
        if True, than frequencies will be returned.
    
    Returns
    ---------------
    > if ret_freqs = True
       * imfs: 2d ndarray,
           intrinsic mode functions and remainder, 
           shape = (order+1,x.shape).           
    > if ret_freqs = True, 
       * freqs: 1d ndarray estimated frequencies.
    Notes
    --------------
    * Maximum estimation frequency fs/4.
    * The frequencies are calculated in radians/2pi 
       (range from 0 to 1), 
       > for calculate it values in Hertz use freqs*fs, 
           where fs is the sampling frequency,
       > for calculate value in points use freqs*N, 
           where N is the sample size,
    * The frequencis are estimated by the spectrum
        barricenter method.
    * Method is sensetive to reularization value, with its
        increasing, imfs bandwidth will growth.   
    * In orginal paper: 
       regularization=2e3; 
       noise_impact=0; 
       max_iter = 500.
    * VMD is calculated as the iterative routine, 
      where all values {IMF_p }_(p=0)^(P=1) 
      and central frequencies {ω_p }_(p=0)^(P=1) 
      are updated on each step as follows [6]:
      ..math::
      IMF_p^(n+1) (ω)=[(f-∑_(i≠p){IMF_i +λ⁄2}])⁄[1+2α(ω-ω_p)^2],
      ω_p^(n+1)=(∫_0^∞{ω|IMF_p(ω)|^2 dω})⁄(∫_0^∞{|IMF_p(ω)|^2 dω}),
      where 
      * IMF_p^n  are spectrum of IMF_p(ω) 
       for p-th component of n-th iteration;
      * ω_p^n  and central frequency ω_p^n
       for p-th component of n-th iteration;
      * α are coefficients corresponded bandwidth;
      * λ are coefficients corresponded to noise influence. 
      The beginning values IMF_p^0 and ω_p^0 
      can be set as zeros or small random values.

    Refernces
    --------------------
    [1] Original paper: 
        Dragomiretskiy, K. and Zosso, D. (2014) 
        "Variational Mode Decomposition", 
        IEEE Transactions on Signal Processing,
        62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
    [2]  Code based on 
    [2a] Dominique Zosso's MATLAB code, available at:
         https://www.mathworks.com/matlabcentral/
         fileexchange/44765-variational-mode-decomposition
    [2b] and Vinícius Rezende Carvalho Python code, available at:
         https://github.com/vrcarva/vmdpy/blob/master/vmdpy/vmdpy.py.
    
    See also
    -------------------
    emd
    hvd
    ewt
    
    '''
    x = np.asarray(x)
    N = x.shape[0]

    N_ext = 2*N
    N2    = N//2 # max freq for estimation = fs/4
    freqs = np.arange(N_ext)/N_ext  #-0.5-(1/N_ext) #TODO: in original paper frequency range is shifted to -fs/2:fs/2
    reularization = reularization*np.ones(order) # does not applied now- for adaptive reularization like in ADAM
    
    #Hilbert transform
    sp_x = 2*np.fft.fft(x,2*N)
    sp_x[N_ext//2:] = 0

    # start with empty noise variance
    err = np.zeros(N_ext, dtype = np.complex)
    
    # matrix of itteration
    imfs     = np.zeros((N_ext, order),dtype=np.complex)  
    imfs_tmp = np.zeros((N_ext, order),dtype=np.complex)  

    #initialization of weights 
    omega = (2/order)*np.arange(order) #insted of 0.5 #may be random

    sum_modes = 0 

    for n in np.arange(max_iter-1): 
        
        #TODO : vectorize this!         
        for k in np.arange(order):
            
            imfs_tmp[:,k] = imfs[:,k]
            
            #accumulate spectrum in cycle (include last element (order-1) for k=0 !!)
            sum_modes += imfs[:,k-1] - imfs[:,k]

            # update spectrum mode by Winner filter with Tikhonov regularization.
            imfs[:,k] = (sp_x - sum_modes + err)/(1+2*reularization[k]*np.square(freqs - omega[k]))
            
            # estimate center frequencies same as LS problem
            # TODO: check if this not optimal, replace with other estimators
            omega[k] = np.sum( freqs[:N2]*np.square( np.abs(imfs[:N2, k]) ) )/np.sum( np.square( np.abs(imfs[:N2,k]) ) )            
        
        if (noise_impact!=None):
            err += noise_impact*(np.sum(imfs[:,:],axis = 1) - sp_x)

        #STOP CRITERIA # variance of changing as square of Frobenius Norms of differnce 
        #TODO: change on the Cauchy criteria!!!
        d_imf = imfs-imfs_tmp
        var = np.sum(np.sum( d_imf * np.conj(d_imf), axis=1 ))/N_ext        
        if(np.abs(var) <= tolerance):
            break    
    
    if(ret_freqs):
        return omega
    
    else:
        #TODO: include residuals?
        out = np.zeros((order, N), dtype=x.dtype)       
        for k in range(order):
            out[k,:] = (np.fft.ifft(imfs[:,k]))[:N]

        return out
    
    
    
# def  vmd_fitzR(x, order, tolerance=1e-7, reularization=2000, max_iter = 500, noise_impact=0, ret_freqs = False):
#     '''
#     Variational mode decomposition.
    
#     The decomposition of the signal on the modes with the requirement 
#         of the most compact spectrum bandwidth around the central frequency of components.

#     Parameters:
#     * x: 1d ndarray.
#     * order: order of the model.
#     * tolerance: variance of recinstructed signal reduction.
#     * reularization: regularization parameter for itterative optimization problem.
#     * max_iter: maximum number of itterations.
#     * noise_impact: noise imact into optimization.
#     * ret_freqs: if True, than frequencies will be returned.
    
#     Returns:
#     * components: 2d ndarray with dimentions = (x.size, order).
#     * if ret_freqs = True: 1d ndarray of frequencies.
    
#     Notes:
#     * Maximum estimation frequency fs/4.
#     * If frequencies will be retuned it value will be in points, for
#         calculate it values in Hertz use freqs*fs, 
#                                 where fs is the sampling frequency.
#     * Method is sensetive to reularization value.    
#     * In orginal paper: regularization=2e3; noise_impact=0; max_iter = 500.
    
#     Refernces:
#     [1] Original paper: 
#         Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’, 
#         IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.
#     [2]  Code based on 
#     [2a] Dominique Zosso's MATLAB code, available at:
#          https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
#     [2b] and Vinícius Rezende Carvalho Python code, available at:
#          https://github.com/vrcarva/vmdpy/blob/master/vmdpy/vmdpy.py
#     '''
#     x = np.asarray(x)
#     N = x.shape[0]

#     N_ext = 2*N
#     N2    = N//2 # max freq for estimation = fs/4
#     freqs = np.arange(N_ext)/N_ext  #-0.5-(1/N_ext) #TODO: in original paper frequency range is shifted
#     reularization = reularization*np.ones(order) # does not applied now- for adaptive reularization like in ADAM
    
#     #Hilbert transform
#     sp_x = 2*np.fft.fft(x,2*N)
#     sp_x[N_ext//2:] = 0

#     # start with empty noise variance
#     err = np.zeros(N_ext, dtype = np.complex)
    
#     # matrix of itteration
#     imfs     = np.zeros((N_ext, order),dtype=np.complex)  
#     imfs_tmp = np.zeros((N_ext, order),dtype=np.complex)  

#     #initialization of weights 
#     omega = (2/order)*np.arange(order) #insted of 0.5 #may be random

#     sum_modes = 0 

#     for n in np.arange(max_iter-1): 
#         #TODO may be replace order?        
#         for k in np.arange(order):
            
#             imfs_tmp[:,k] = imfs[:,k]
            
#             #accumulate spectrum in cycle (include last element (order-1) for k=0 !!)
#             sum_modes += imfs[:,k-1] - imfs_tmp[:,k]

#             # update spectrum mode by Winner filter with Tikhonov regularization.
#             imfs[:,k] = (sp_x - sum_modes + err)/(1+2*reularization[k]*np.square(freqs - omega[k]))
            
#             # estimate center frequencies same as LS problem
#             # TODO: check if this not optimal, replace with other estimators
#             omega[k] = estimators.cdf.MCRB(#np.sum( freqs[:N2]*np.square( np.abs(imfs[:N2, k]) ) )/np.sum( np.square( np.abs(imfs[:N2,k]) ) )            
        
#         if (noise_impact!=None):
#             err += noise_impact*(np.sum(imfs[:,:],axis = 1) - sp_x)

#         #STOP CRITERIA # variance of changing as square of Frobenius Norms of differnce 
#         d_imf = imfs-imfs_tmp
#         var = np.sum(np.sum( d_imf * np.conj(d_imf), axis=1 ))/N_ext        
#         if(np.abs(var) <= tolerance):break    
    
#     if(ret_freqs):
#         return omega
    
#     else:
#         #TODO: include residuals?
#         out = np.zeros((N, order), dtype=x.dtype)       
#         for k in range(order):
#             out[:,k] = (np.fft.ifft(imfs[:,k]))[:N]

#         return out