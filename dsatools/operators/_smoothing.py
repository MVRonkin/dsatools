import numpy as np

from ._lags_matrix import conv_matrix

#TODO: merge both implementations
# probably for this padded signal need to be with mean value.
__all__=['moving_average','movav']

def moving_average(vector,av_cof, mode='postw'):
    '''
    Simple Moving Average or Weighted  Moving Average
    
    Parameters
    -------------
    * vector: 1d ndarray,
        input vector.
    * av_cof: int or 1d ndarray,
        > if av_cof is int:averaging-window size;
        > if av_cof is 1d ndarray:weights array 
                        for weighted moving average;
    * mode: string,
        mode = {'full',prew','postw','valid','same'} 
        > mode = full: output with size = av_cof+N-1
        > mode = prew: output with size = N, 
            padding of zeros in the first.
        > mode = postw: output with size = N,     
             padding of zeros in the last.
        > mode = valid: output with size = N-av_cof+1,     
             without padding of zeros.  
        > mode = same: output with size = N,     
             without padding of zeros at the begin and end. 
     Returns
    -----------
    * smoothed array: 1d ndarray.          
    '''
    av_cof = np.asarray(av_cof,dtype=int)
    
    if av_cof.size ==1:
        window = np.ones(av_cof)
    else:
        window = av_cof
        
    vector = np.asarray(vector)
    
    return conv_matrix(vector,mode=mode,lags=window.size) @ window/window.size
#---------------------------------------------------
def movav(vector,window_size,straight = True):
    '''
    Alternative Moving Average Implementation.
    
    Paramters
    -----------------
    vector: 1d ndarray,
        input vector.
    window_size: size of sliding window.
    straight: bool,
        if true, sliding in straight direction,
        if false, sliding in backward direction.
    
    Return
    ----------
    smoothed vector: 1d ndarray.
    '''
    vector = np.asarray(vector)
    N = vector.shape[0]
    out = np.zeros_like(vector)
    
    if straight:
#         if symmetry:
        for i in range(N):
            lp = min(N,i+window_size//2)
            fp = max(i-window_size - window_size//2,0)
            out[i] = np.mean(vector[fp:lp])
#         else:
#             for i in range(N):
#                 lp = min(N,i+size)
#                 out[i] = np.mean(vector[i:lp])
    else:
        for i in range(N-1,-1,-1):
            fp = N-min(N-1,i+window_size//2)-1
            lp = N-max(i-window_size - window_size//2,0)-1
            out[N-i-1] = np.mean(vector[fp:lp])
#         else:
#             for i in range(N,0,-1):
#                 lp = N-min(N,i+size)+1
#                 out[i] = np.mean(vector[N-i+1:lp]) 
    return out
