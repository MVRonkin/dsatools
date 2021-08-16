import numpy as np

from ._lags_matrix import conv_matrix

__all__=['moving_average',]

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
