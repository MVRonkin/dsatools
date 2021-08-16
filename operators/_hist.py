import numpy as np
import scipy


__all__ = ['hist']

__EPSILON__ = 1e-8

#--------------------------------------------------------------------
def hist(x,y=None,n_bins = None, bins = None, normalize = False):
    '''
    Historgram of inputs made by 
      the set number of bins (n_bins) or by set of bins.        
    
    Parameters
    -----------
   * x,y: 1d ndarrays.
   * n_bins:int or None,
       required number of uniformly distributed bins
       (n_bins work only if bins is None).
   * bins: 1d ndarray or None,
        grid of prepared bins (can be ununiform).
   * normalize: bool,
       if True, historgram will be normalized on sum of it values.
   
   Returns
   ---------
   * If y is not None ->(out_x, out_y,bins) 
   * If y is None     ->(out_x,bins) 
        
   Notes
   --------
   * In differ with numpy implementation 
       the reurned nubmer of bins the same 
       as size of histogram array. 
       For obtaining the same bins grid 
       as in numpy bins from 1 have to be taken.
   * n_bins is taken into account if only bins = None. 
   * If bins is set and n_bins is set, 
     than hist will be taken by bins (priorety higher).
   * If n_bins is None or n_bins<1 and bins is None: 
     n_bins = x.shape[0]//10
   * Special cases n_bins = 'xy': 
       than bins will be taken as 
       sorting concotenated x and y.
   * Special cases n_bins = 'xy10': 
      than bins will be taken as each tens of 
      sorting concotenated x and y . 

    '''
    x = np.asarray(x)
    if y is not None: 
        y = np.asarray(y)
        if (y.shape != x.shape): raise ValueError('y.shape ! = x.shape')    
    
    if(bins is None):       
        bins = take_bins(x,y, n_bins)
    
    else:
        bins = np.asarray(bins)
    
    if(y is None):
        _,out_x = _hist_(x, n_bins = n_bins, bins = bins, 
                         add_zero_bin = False, normalize=normalize)
        out = (out_x, bins)

    else:
        _,out_x = _hist_(x, n_bins = None, bins = bins, 
                         add_zero_bin = False, normalize=normalize)
        _,out_y = _hist_(y, n_bins = None, bins = bins, 
                         add_zero_bin = False, normalize=normalize)
        out = (out_x,out_y, bins)
        
    return out 


__BIN_SCALE__ = 10

def take_bins(x,y=None, n_bins=None):
    ''' 
    Auxilary function for build the set of bins.
    
    Parameters
    ------------
   * x,y 1d ndarrays
   * n_bins: int or None,
       required number of uniformly distributed 
       bins work only if bins is None.
    
    Returns
    ----------
    * bins: 1d ndarray.
        
   Notes
   ----------
   * If y is nont None, 
       bins will be taken on the joint x and y. 
   * If n_bins is None or n_bins<1 and bins is None: 
            n_bins = x.shape[0]//10
   * Special cases n_bins = 'xy': 
       than bins will be taken 
       as sorting concotenated x and y.
    * Special cases n_bins = 'xy10': 
        than bins will be taken as each tens 
        of sorting concotenated x and y.        
        
    '''
    x = np.asarray(x)

    # ununiform case
    if(n_bins in ['xy','xy10']):
        if(y is None):
            out= np.sort(x)
        else:
            y = np.asarray(y)
            out = np.sort(np.concatenate(( x,y )) )
        if(n_bins=='xy10'):
            out = out[::10]
        return out
   
    # uniform case
    if(n_bins is None or n_bins<1):
        n_bins = x.shape[0]//__BIN_SCALE__
    
    if(n_bins<1): # if still n_bins<1
        raise ValueError('shape have to be such that n_bins 1 or greater')
    
    if(y is None):
        hist_max = np.max(x)
        hist_min = np.min(x)
    else:
        hist_max = np.max([np.max(x), np.max(y)])
        hist_min = np.min([np.min(x), np.min(y)])

    hist_step  = (hist_max - hist_min)/(n_bins)
    
    bins = hist_step*np.arange(1,n_bins+1)+hist_min 
    
    return bins 

#--------------------------------------------------------------------
def _hist_(x,n_bins = None, bins = None, add_zero_bin = False, normalize = False):
    '''
    Function for histogram construction.
    
    Parameters
    ------------
    * x,y: 1d ndarrays
    * n_bins: required number of uniformly 
        distributed bins work only 
        if bins is None.
    * bins: 1d ndarray,
        grid of prepared bins (can be ununiform).
    * normalize: bool,
        If True, the historgram will be normalized 
        on sum of it values.
    * add_zero_bin:
        if true return 0 as number of bins befor 
        first bins for make it the same as in numpy.
    
    Returns
    ----------
    * out_x: 1d ndarray.
    * bins: 1d ndarray. 

    '''
    x = np.asarray(x)
    
    hist_x = np.sort(x)    
    n_bins = bins.shape[0]
    
    out    = np.zeros(n_bins)
    out[0] = hist_x[hist_x<bins[0]].size
    last   = out[0]    
    for i in np.arange(1,n_bins):
        cur    = hist_x[hist_x<bins[i]].size
        out[i] = cur-last
        last   = cur

    out[-1] += x.size - np.sum(out)

    if(add_zero_bin):
        bins, out = np.append([hist_x[0]],bins), np.append([0],out)
    
    if(normalize):
        out /=np.sum(out)

    return bins, out

#--------------------------------------------------------------------
def __check_bins__(bins, hist_x, hist_y=None ):
    ''' check on shape and type'''
    hist_x = np.asarray(hist_x)
    bins   = np.asarray(bins)
    
    if(hist_y is not None):
        hist_y = np.asarray(hist_y)   
    
        if (hist_x.shape != hist_y.shape ):
            raise ValueError('hist_x.shape != hist_y.shape')
            
    if (hist_x.shape != bins.shape ):
        raise ValueError('hist_x.shape != bins.shape') 
    
    return bins, hist_x, hist_y 

#--------------------------------------------------------------------

# #--------------------------------------------------------------------  
# def join_hists(hist_x, hist_y):
#     ''' 
#         Return the joint histograms 
#         (the same as hist_x + hist_y - cross_hists) 
        
#         :param: hist_x,hist_y 1d histograms of the joint bins grid (ndarrays)
#         :return: hist (joiny hist)
        
#     '''
#     hist_x, hist_y  = axut.check_xy(hist_x, hist_y,mode = 'None',take_mean= False, epsilon = 0 )
#     return np.max(np.vstack((hist_x, hist_y)),axis=0)
# #     return np.asarray([np.max([hist_x[i],hist_y[i]]) for i in np.arange(hist_x.shape[0]) ])

# #--------------------------------------------------------------------  
# def cross_hists(hist_x, hist_y):
#     ''' 
#         Return the cross of histograms 
#         (the same as hist_x + hist_y - join_hists) 
        
#         :param: hist_x,hist_y 1d histograms of the joint bins grid (ndarrays)
#         :return: hist (joiny hist)
        
#     '''
#     hist_x, hist_y  = axut.check_xy(hist_x, hist_y,mode = 'None',take_mean= False, epsilon = 0 )
#     return np.max(np.vstack((hist_x, hist_y)),axis=0)
# #     return np.asarray([np.min([hist_x[i],hist_y[i]]) for i in np.arange(hist_x.shape[0]) ])

# #--------------------------------------------------------------------  
# def hist2cdf(hist_x, normalize = True):
#     ''' 
#         Return the cumulative density function made by histogram.
#         :param: hist_x 1d histogram (ndarray)
#         :return: cfd (Cumulative Density Function)        
#     '''
#     hist_x = np.asarray(hist_x)
    
#     out = np.cumsum(hist_x)
    
#     if(normalize):
#         out /=np.max(out)
# #   TODO:      out /=x.size # more simple!
#     return out
# #-------------------------------------------------------------------- 
# def cdf_by_hist(x,y=None,n_bins = None, bins = None, take_mean=False):
#     ''' 
#         Cumulative density function constructed by histogram.
        
#         :param: x,y 1d ndarrays
#         :param: n_bins required number of uniformly distributed bins
#                      work only if bins is None
#         :param: bins - grid of prepared bins (can be ununiform)
#         :param: take_mean sustrauct mean if ture
        
#         :return: y is not None ->  (out_x, out_y,bins) 
#         :return: y is None     ->  (out_x,bins) 
        
#         Note:
#         if bins is None and n_bins is None: 
#             bins = np.sort(np.concatenate((x,y)))
#             This case make the same result as ecdf!

#         if bins is None and n_bins <=0:
#             n_bins = x.shape[0] 
#             The case of uniform bins grid! (Differ from ECDF)
            
#         for tests: modes n_bins = 't10' and n_bins = 't5' 
#             for obtaining uniform bins with x shape/10 and /5 correspondingly
            
#     '''
#     #FIXME: the results are sligthly differ from ecdf
#     # TODO: the case xy is the same as for ecfd, but uniform bins may be more valid (see tests)
#     if(bins is None and n_bins is None):       
#         bins = take_bins(x,y, n_bins='xy')
    
#     elif(n_bins == 't10' and bins is None):
#         bins = take_bins(x,y, n_bins=x.shape[0]//10)
        
#     elif(n_bins == 't5' and bins is None):
#         bins = take_bins(x,y, n_bins=x.shape[0]//5)        

#     if(y is None):
#         bins, out_x = hist(x,y=None,n_bins = n_bins, bins = bins, take_mean=take_mean)
#         out_x = hist2cdf(out_x, normalize = True)
#         out   = (bins, out_x )
        
#     else:
#         bins, out_x, out_y = hist(x,y=y,n_bins = n_bins, bins = bins, take_mean=take_mean)
#         out_x = hist2cdf(out_x, normalize = True)
#         out_y = hist2cdf(out_y, normalize = True)        
#         out   = (bins,out_x, out_y)
    
#     return out

#--------------------------------------------------------------------
# def ecdf(x,y=None):
#     ''' 
#         Empirical Cumulative Density Function (ECDF),
        
#         Note: * Based on scipy implementation.        
#               * If y is not None, ECDF will be constructed on the
#                             joint x and y.
#               * If y is None, only bins and cdf(x) (2 argument) will be
#                           returned
        
#         :param: x,y 1d ndarrays    
        
#         :return: 
#                  y is not None ->  (out_x, out_y,bins) 
#                  y is None     ->  (out_x,bins) 
       
#     '''
#     #TODO: cdf by hist slightly shift with respect to ecdf
#     x = np.array(x)
#     x = np.sort(x)
    
#     ret2 =True
#     if (y is not None):
#         y = np.array(y)
#         y = np.sort(y)
#     else:
#         ret2 = False
#         y=np.array([])
        
#     bins = np.concatenate((x,y))
#     bins=np.sort(bins)
#     x_cdf = np.searchsorted(x,bins, 'right')
#     y_cdf = np.searchsorted(y,bins, 'right')
#     x_cdf = (x_cdf) / x.shape[0]    
#     y_cdf = (y_cdf) / y.shape[0]
    
#     out = (bins,x_cdf)
    
#     if (ret2):
#         out= (bins,x_cdf,y_cdf)

#     return out

#--------------------------------------------------------------------
