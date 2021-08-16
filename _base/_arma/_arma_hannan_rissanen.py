import numpy as np
import scipy  

from ... import operators
from ... import utilits as ut

from . _ar_yule_walker import ar_yule_walker

__all__ = ['arma_hannan_rissanen']
#------------------------------------------------------------------
def arma_hannan_rissanen(x, poles_order=0, zeros_order=0, unbias = True):
    '''    
    Hannan_Rissanen method for
        autoregressive - moving average 
        (ARMA) model approximation.

    Parameters
    ---------------
    * x:  1d ndarray. 
    * poles_order: int.
         the autoregressive model (pole model) 
         order of the desired model. 
    * zeros_order: int.
         the moving average model (zeros model) 
         order of the desired model.          
    * unbias: bool,
        if True, unbiased autocorrleation 
            (sum(x(k)*x(n-k))/(N-n)) will be taken.
      
    Returns
    --------------
    * a: 1d ndarray,
        autoregressive coefficients of the ARMA model.
    * b: 1d ndarray,
        moving average coefficients of the ARMA model.        
    * noise_variace: complex of float,
        variance of model residulas.
       
    Notes: 
    ------------
    * Here are implemented simplified model.
        High order AR model is taken equal to
        deisred one.

    Examples
    ------------
    
    References
    ------------
    [1] Brockwell, Peter J., and Richard A. Davis. 2016.
       Introduction to Time Series and Forecasting. Springer.
    
    See also
    -----------

    '''    
    x = np.asarray(x)
    N = x.shape[0]
    
    a,_ = ar_yule_walker(x,
                         poles_order,
                         unbias=unbias)
    
    r = operators.lags_matrix(x,
                              mode='full',
                              lags=poles_order+1,)
    r1 = r[zeros_order:,0] #x[poly_order+zreo_order]
    
#     for i in range(1):
    #------------    
    resid = r[:,0] - r[:,1:].dot(-a[1:])

    rresid = operators.lags_matrix(resid,
                                   mode='full',
                                   lags=zeros_order+1,) 

    rn = np.append(r[zeros_order:,1:], 
                   rresid[2*zeros_order:,1:],axis=1)

#     res=np.dot(np.linalg.pinv(-rn),r1)
    res = scipy.linalg.lstsq(rn,r1)[0]
    a = np.append([1],-res[:poles_order])
    #------------ 
    
    b = res[poles_order:]#np.append([0],res[poles_order:])
    
    err=1

    return a,b,err
   

   
# def arma_hannan_rissanen_unbiased(x, poly_order=0, zero_order=0, 
#                                       unbias = True, n_psd = None):
#     '''  
#     #FOR TEST!
#     Hannan_Rissanen method for autoregressive - moving average 
#       (ARMA) model approximation with additinal unbias of coefficients.

#     Parameters
#     ---------------
#     * x:  1d ndarray,
#                 inputs. 
#     * poly_order: int.
#          the autoregressive model (pole model) 
#          order of the desired model. 
#     * zero_order: int.
#          the moving average model (zeros model) 
#          order of the desired model.          
#     * n_psd: int or None.
#         length of desired pseudospctrum, 
#         if None, n_psd = x.shape[0],
#         if n_psd<0, than model coefficients (1,-a) 
#           and noise_variance (\sigma^2) will be returend.
#     * unbias: bool,
#         if True, unbiased autocorrleation 
#             (sum(x(k)*x(n-k))/(N-n)) will be taken.
      
#     Returns
#     --------------
#     > if n_psd>0: 
#         * pseudo-spectrum,
#     > else: 
#        * ar_cofs (a), ma_cofs (b) - 2 1d ndarray; 
#        * noise_variace - variance of model residulas.
       
#     Notes: 
#     ------------
#     * Here are implemented simplified model.
#         High order AR model is taken equal to
#         deisred one.

#     Examples
#     ------------
    
#     References
#     ------------
#     [1] Brockwell, Peter J., and Richard A. Davis. 2016.
#        Introduction to Time Series and Forecasting. Springer.
    
#     See also
#     -----------

#     '''    
#     x = np.asarray(x)
#     N = x.shape[0]

#     a,b,_ = arma_hannan_rissanen(x, 
#                                  poly_order=poly_order, 
#                                  zero_order=zero_order, 
#                                  unbias    = unbias, 
#                                  n_psd     = -1)
#     # unbias
#     z = np.zeros(x.shape,dtype = x.dtype)    
#     for n in np.arange(np.max([poly_order, zero_order]), N):
#         tmp_ar = np.dot(-a[1:], x[n - poly_order:n][::-1])
#         tmp_ma = np.dot(b,x[n - zero_order:n][::-1])
#         z[n] = x[n] - tmp_ar - tmp_ma

#     mh = scipy.signal.lfilter([1], a, z)
#     ah = scipy.signal.lfilter(np.r_[1, -b], [1], z)
#     #i'm not sure here
#     rm = matrix.lags_matrix(mh,
#                            mode='full',
#                            mcolumns=poly_order+1,)[2*poly_order:,:-1]
    
    
#     ra = matrix.lags_matrix(ah,
#                             mode='full',
#                             mcolumns=zero_order+1,)[2*zero_order:,:-1]
#     print(ra.shape,rm.shape)
#     r1 = z[max(poly_order, zero_order):] #x[poly_order+zreo_order]
    
#     rn = np.append(rm[max(zero_order - poly_order, 0):,:], 
#                    ra[max(poly_order - zero_order, 0):,:],axis=1)

#     res=np.dot(np.linalg.pinv(rn),r1)
    
#     err = np.sum(np.square(r1- rn.dot(res)))/res.size
    
#     a = np.append([1],-(-a[1:]+res[:poly_order]))
#     b = b+res[poly_order:]

#     if(n_psd<1):
#         return a,b,err
#     else:
#         psd = ut.arma2psd(a,b,np.abs(err),n_psd)
#         return psd


# def arma_hannan_rissanen(x, poly_order=0, zero_order=0, 
#                                  unbias = True, n_psd = None):
#     '''    
#     Hannan_Rissanen method for autoregressive - moving average 
#                                     (ARMA) model approximation.

#     Parameters
#     ---------------
#     * x:  1d ndarray,
#                 inputs. 
#     * poly_order: int.
#          the autoregressive model (pole model) 
#          order of the desired model. 
#     * zero_order: int.
#          the moving average model (zeros model) 
#          order of the desired model.          
#     * n_psd: int or None.
#         length of desired pseudospctrum, 
#         if None, n_psd = x.shape[0],
#         if n_psd<0, than model coefficients (1,-a) 
#           and noise_variance (\sigma^2) will be returend.
#     * unbias: bool,
#         if True, unbiased autocorrleation 
#             (sum(x(k)*x(n-k))/(N-n)) will be taken.
      
#     Returns
#     --------------
#     > if n_psd>0: 
#         * pseudo-spectrum,
#     > else: 
#        * ar_cofs (a), ma_cofs (b) - 2 1d ndarray; 
#        * noise_variace - variance of model residulas.
       
#     Notes: 
#     ------------
#     * Here are implemented simplified model.
#         High order AR model is taken equal to
#         deisred one.

#     Examples
#     ------------
    
#     References
#     ------------
#     [1] Brockwell, Peter J., and Richard A. Davis. 2016.
#        Introduction to Time Series and Forecasting. Springer.
#     See also
#     -----------

#     '''    
#     x = np.asarray(x)
#     N = x.shape[0]
    
#     if n_psd == None: n_psd = N
    
#     a,_ = spectrum.yule_walker(x,
#                                poly_order,
#                                n_psd=-1,
#                                unbias=unbias)
#     a = -a[1:]
    
#     r = matrix.lags_matrix(x,
#                            mode='full',
#                            mcolumns=poly_order+1,)
    
#     resid = r[:,0] - r[:,1:].dot(a)

#     rresid = matrix.lags_matrix(resid,
#                                 mode='full',
#                                 mcolumns=zero_order+1,)
    
# # Alternatively covar mode can be applied    
# #     r = matrix.lags_matrix(x,
# #                            mode='covar',
# #                            mcolumns=poly_order+1,)
    
# #     resid = r[:,0] - r[:,1:].dot(a)

# #     rresid = matrix.lags_matrix(resid,
# #                                 mode='covar',
# #                                 mcolumns=zero_order+1,)
# # rn = np.append(r[zero_order:,1:], rresid[:,1:],axis=1)

#     r1 = r[zero_order:,0] #x[poly_order+zreo_order]
    
#     rn = np.append(r[zero_order:,1:], rresid[2*zero_order:,1:],axis=1)

#     res=np.dot(np.linalg.pinv(-rn),r1)
    
#     a = np.append([1],res[:poly_order])
    
#     b = res[poly_order:]
    
#     err=1
#     if(n_psd<1):
#         return a,b,err
#     else:
#         psd = ut.arma2psd(a,b,np.abs(err),n_psd)
#         return psd