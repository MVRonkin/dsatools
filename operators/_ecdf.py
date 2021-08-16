import numpy as np
import scipy

from ._hist import take_bins

__all__ = ['ecdf']

__EPSILON__ = 1e-8
#--------------------------------------------------------------------
def ecdf(x,y=None):
    ''' 
    Empirical Cumulative Density Function (ECDF).

    Parameters
    -----------
    * x,y: 1d ndarrays,
        if y is None, than ecdf only by x will be taken.

    Returns
    --------
    * if y is not None ->  (bins,out_x, out_y); 
    * if y is None     ->  (bins,out_x). 

    Notes
    -------
    * Based on scipy implementation.        
    * If y is not None, ECDF will be constructed on the joint x and y.
    * If y is None, only bins and cdf(x) (2 argument) will be returned.
    * ECDF is calculated as:    
        bins  = sort(concatenate(x,y)),
        cdf_x = (serch&past bins in sort(x))/size(x),
        cdf_y = (serch&past bins in sort(y))/size(y),
        where:
        * bins - bins for cdfs (if y is not None, joint bins).  
    '''
    x = np.array(x)
    x = np.sort(x)
    
    ret2 =True
    if (y is not None):
        y = np.array(y)
        y = np.sort(y)
    else:
        ret2 = False
        y=np.array([])
        
    bins = np.concatenate((x,y))
    bins=np.sort(bins)
    x_cdf = np.searchsorted(x,bins, 'right')
    y_cdf = np.searchsorted(y,bins, 'right')
    x_cdf = (x_cdf) / x.shape[0]    
    y_cdf = (y_cdf) / y.shape[0]
    
    out = (bins,x_cdf)
    
    if (ret2):
        out= (bins,x_cdf,y_cdf)

    return out
#--------------------------------------------------------------------  
def hist2cdf(hist_x, normalize = True):
    ''' 
    The cumulative density function made by histogram.
    
    Parameters:
      * hist_x 1d histogram (ndarray).
    
    Returns:
      * cfd(hist_x) (Cumulative Density Function).        
    '''
    hist_x = np.asarray(hist_x)
    
    out = np.cumsum(hist_x)
    
    if(normalize):
        out /=np.max(out)
#   TODO:      out /=x.size # more simple!
    return out
#-------------------------------------------------------------------- 
def cdf_by_hist(x,y=None,n_bins = None, bins = None, take_mean=False):
    ''' 
    Cumulative density function constructed by histogram.
    
    Parameters:    
      * x,y: 1d ndarrays;
      * n_bins: required number of uniformly distributed bins,
                  * work only if bins is None.
      * bins: grid of prepared bins (can be ununiform)
      * take_mean: sustrauct mean if ture.
    
    Returns:    
      * y is not None ->  (out_x, out_y,bins) 
      * y is None     ->  (out_x,bins) 
        
    Notes:
      * If bins is None and n_bins is None: 
            bins = np.sort(np.concatenate((x,y))).
            This case make the same result as ecdf!

      * If bins is None and n_bins <=0: n_bins = x.shape[0]; 
            The case of uniform bins grid! (Differ from ECDF).
            
      * For tests: modes n_bins = 't10' and n_bins = 't5' 
            for obtaining uniform bins with x shape/10 and /5 correspondingly
            
    '''
    #FIXME: the results are sligthly differ from ecdf
    # TODO: the case xy is the same as for ecfd, but uniform bins may be more valid (see tests)
    if(bins is None and n_bins is None):       
        bins = take_bins(x,y, n_bins='xy')
    
    elif(n_bins == 't10' and bins is None):
        bins = take_bins(x,y, n_bins=x.shape[0]//10)
        
    elif(n_bins == 't5' and bins is None):
        bins = take_bins(x,y, n_bins=x.shape[0]//5)        

    if(y is None):
        bins, out_x = hist(x,y=None,n_bins = n_bins, bins = bins, take_mean=take_mean)
        out_x = hist2cdf(out_x, normalize = True)
        out   = (bins, out_x )
        
    else:
        bins, out_x, out_y = hist(x,y=y,n_bins = n_bins, bins = bins, take_mean=take_mean)
        out_x = hist2cdf(out_x, normalize = True)
        out_y = hist2cdf(out_y, normalize = True)        
        out   = (bins,out_x, out_y)
    
    return out

