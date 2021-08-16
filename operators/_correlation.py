import numpy as np
import scipy

#--------------------------------------------------------
__all__ = ['autocorrelation','correlation','convolution']
#--------------------------------------------------------
__EPSILON__   = 1e-6

#--------------------------------------------------------------    
def correlation(x,y = None,  mode='straight', take_mean = False, unbias = False):  
    '''
    Correlation function:
    ::math..
    R[i] = sum_n(x[n+i]*conj(y[n])).
    
    Parameters
    ------------    
    * x, y: 1d ndarrays,
        inputs 1d array (ndarray)
        if y is None - autocorrelation function is taken. 
    * mode: string,
        mode = {'full','same','None','straight'}:
      * 'straight': 
        (by default) straight direction (size of model N),
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N)[:N], 
        size of model N.
      * 'full': 
        size of output 2N-1 (bith directions): 
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N), 
        size of output 2N-1.
      * 'same': 
        the same size of output as input (N):  
        R = R[N//2-1:-N//2], size of model N lags from -N/2 to N/2.                        
      * 'None': 
          return input x (FOR TESTS).
    * take_mean: bool,
        if True, mean values of x and y substructed first. 
    * unbias:  bool,  
        if True,  R = {R[n]/(N-|n|)}_n^N-1.    
    
    Returns
    ------------
    x: 1d ndarray,
     correlation function.
    
    Notes
    -------------
    * Basicly the correlation is calculated above two derections 
        (in the straight and backward direction), 
        consequencely it is taken for double size of samples (full mode).   
    * Here correlation function is calculated using 
        fast Fourier transform,
        thus more correctly to say:
        R = ifft( fft(x, 2N)* conj( fft(y, 2N) ) ) [N+1:],[:N],     
        where first part is backward part, 
        and the second part is straightforward.
                         
    '''    

    x = np.asarray(x)
    
    if mode is 'None':
         return x
        
    if y is None: y = x
    else: 
        y = np.asarray(y)
        if (y.shape != x.shape): raise ValueError('y.shape ! = x.shape')    
    
    N  = x.shape[0]
    Sp = np.fft.fft(x,2*N)*np.conj((np.fft.fft(y,2*N)))
    R  = np.fft.ifft(Sp)      

# TODO: what does it mean FB for correlation matrix:    
#     if(FB):
#         Sp = np.fft.fft(x[::-1],2*N)*np.conj((np.fft.fft(y[::-1],2*N)))   
#         R2 = np.fft.ifft(Sp)[::-1] 
#         R  = (R+R2)/2
        
    if(unbias):        
        R[:N]  /=(N-np.arange(N)) 
        R[N+1:]/=(np.arange(N-1)+1)
    else:
        R[:N]  /=(N) 
        R[N+1:]/=(N-1)        

    if(mode is 'full'):#ifftshift
        R = np.append(R[N+1:],R[:N])
    
    elif(mode is 'straight'):
        R = R[:N]        
    
    elif(mode is 'same'):
        R = np.append(R[N+N//2:],R[:N//2])
    else:
        raise NotImplementedError('use one of the aviliable modes')
        
    return  R  

#--------------------------------------------------------------
def autocorrelation(s,N=None):    
    '''
    Autocorrelation function.
    
    FOR COMPATIBILITY.
    Parameters
    ------------
    * x: 1d ndarray
        inputs 1d array.
    * N: int or None,
        need as heritage for old-style functions.
    
    Returns
    ------------
    * x: 1d ndarray,
        autocorrelation function.      
                
    Notes
    --------------
    * the function is same as correlation(x,x,mode = 'straight'),
       in fact function is calculated as ifft(|fft(x,2N)|^2)[:N], 
       where N is the signal length.
    *  Autocorrelation function:
        c[i] = np.sum_n(x[i]*conj(x[i-n])).

    '''
    s = np.asarray(s)
    if(N is None): N = s.shape[0]
    
    Sp    = np.fft.fft(s,2*N)
    R     = np.fft.ifft(Sp*np.conj(Sp))      
    
    R     = R[:N]                   
    
    return  R

#--------------------------------------------------------------
def xycor(s1,s2, mode = 'R12', modecor='same' ):
    '''
    Function for special cross-correlation modes.
    
    Parameters
    -------------
    * s1,s2: 1d ndarrays 
        input signals.
    * mode: string,
        cross-correlation modes
        mode={'R12','R21','Rfb','Rfb12','Rfb21'}.
    * modecor: string,
        mode of correlation function,
        modecor={'same','full','straight'}.
    
    Returns
    ----------
    *R_x,R_y: 1d ndarrays,
        output arrays, depends on mode:    
        > R12: Rf, R1.
        > R21: Rb, R2.
        > Rfb: Rf, Rb.
        > Rfb12: Rf*R1^*, Rb*R2^*.
        > Rfb21: Rf*R2^*, Rb*R1^*.
    
    Notes
    --------
    * There are following notations are use:
        Rf = s2 \cdot s1^*;
        Rb = s1 \cdot s2^*;
        R1 = s1 \cdot s1^*;
        R2 = s2 \cdot s2^*;
        where \cdot denote convolution operation.

    '''    
    unbias = False
    if(mode in ['R12','None',None]):
        R_x  = correlation(s2,s1, mode=modecor, 
                    take_mean = False, unbias =unbias) 
        R_y  = correlation(s1,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
    
    elif(mode == 'R21'):
        R_x   = correlation(s1,s2, mode=modecor, 
                    take_mean = False, unbias =unbias)    
        R_y  = correlation(s2,s2, mode=modecor, 
                    take_mean = False, unbias =unbias)
        
    elif(mode == 'Rfb'):
        R_x  = correlation(s1,s2, mode=modecor, 
                    take_mean = False, unbias =unbias)    
        R_y  = correlation(s2,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
        
    elif(mode == 'Rfb12'):    
        Rf  = correlation(s1,s2, mode=modecor, 
                    take_mean = False, unbias =unbias) 
        Rb  = correlation(s2,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
        R1  = correlation(s1,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
        R2  = correlation(s2,s2, mode=modecor, 
                    take_mean = False, unbias =unbias)
        R_x = Rf*np.conj(R1)
        R_y = Rb*np.conj(R2)
    
    elif(mode == 'Rfb21'):    
        Rf  = correlation(s1,s2, mode=modecor, 
                    take_mean = False, unbias =unbias) 
        Rb  = correlation(s2,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
        R1  = correlation(s1,s1, mode=modecor, 
                    take_mean = False, unbias =unbias)
        R2  = correlation(s2,s2, mode=modecor, 
                    take_mean = False, unbias =unbias)       
        R_x = Rf*np.conj(R2)
        R_y = Rb*np.conj(R1)
    
    return R_x ,R_y

#--------------------------------------------------------------
def convolution(x,y,  mode='straight'):
    '''
    Convolution function:
    ::math..        
     c[i] = sum_n(x[i]*conj(y[i-n])),
        which is the same as correlation(x,y[::-1]).

    Parameters
    ------------ 
    * x, y: inputs 1d array (ndarray)
        if y is None - autocorrelation function is taken. 
    * mode: ['xcorr','full','same','None','straight']|:
      * 'None','straight' or 'xcorr': 
        (by default) straight direction (size of model N),
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N)[:N], 
            size of model N.
      * 'full': 
        size of output 2N-1 (bith directions): 
        R = ifft(fft(x, 2N)*conj(fft(y, 2N)), 2N), 
        size of output 2N-1.
      * 'same': 
        the same size of output as input (N):  
        R = R[N//2-1:-N//2], size of model N lags from -N/2 to N/2.

    Returns
    -----------
    * 1d array convolution function.
                
    Notes
    -----------------
    * Basicly convolution in formule above two derections (in the straight 
        and backward direction) function, consequencely it is taken for double 
        size of samples (full mode), if  mode = straight, only 
        the first half part of correlation is taken.                       
    * Correlation function allow also same mode but it does 
                                not reccomended for this function.                    
    '''
    #TODO: Add linear and circular convolutions!
    # Add convolution by datamtx
    return correlation(x,y[::-1],  mode=mode, take_mean = False, unbias = False)
#--------------------------------------------------------------
# def crossCorr_clear(x1, x2, nlags=-1):    
#     ''' 
#         Does not applied now, for test
#         old-style function 
#             need to be improved
#         same as np.asarray([np.sum(x1[i:]*np.conj(x2[:N-i])) for i in np.arange(N)])
#     '''
#     r = [np.sum(x1*np.conj(x2))]
#     for i in range(1,len(x1)):
#         r += [np.sum(x1[i:]*np.conj(x2[:-i]))]
#     return np.array(r)
#--------------------------------------------------------------


# # Does not applied Now due to many modes (unconvinient to use)
# __CROSS_MODIFED_MODES__ = ['',None,'None','R12','R21','Rfb','Rfb12','Rfb21']

# def cross_modified_correlation(x,y, take_mean = True,unbias = True, FB = True, cormode = 'full', crossmode='None'):
#     '''
#         Does not applied now, for test
        
#         crossmode = ['None','R12','R21','Rfb','Rfb12','Rfb21']
#         cormode   = ['None','valid','xcorr','full','same', 'argmax']
#     '''
#     x = np.asarray(x)
#     N=x.shape[0]
    
#     if(not Nlags):
#         Nlags = N//2
        
#     if(y is None):
#         y = x
#     else:
#         y = np.asarray(x)
#         if(x.shape != y.shape):
#             raise ValueError('x.shape != y.shape')

#     if(crossmode=='R12'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R12*np.conj(R11)
        
#     elif(mode=='R21'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R21*np.conj(R11) 
        
#     elif(mode=='Rfb'):
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R21*np.conj(R12)
        
#     elif(mode=='Rfb12'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R22 = correlation(y,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R1 = R12*np.conj(R11)    
#         R2 = R21*np.conj(R22)
#         R = R1*np.conj(R2)
        
#     elif(mode=='Rfb21'):
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R22 = correlation(y,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R21 = correlation(y,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R1 = R12*np.conj(R22)    
#         R2 = R21*np.conj(R11)
#         R = R1*np.conj(R2)  
        
#     else:
#         R11 = correlation(x,x, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R12 = correlation(x,y, mode=cormode,take_mean = take_mean, unbias = unbias, FB = FB)
#         R = R12*np.conj(R11)
    
#     return R

    
    
    
    
    
    
    
    
