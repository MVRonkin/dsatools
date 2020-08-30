import numpy as np
import scipy

__all__ = ['p_variaton']

__EPSILON__ = 1e-8

def p_variaton(x,y=None,p=2,take_mean = False):    
    ''' 
    P-variation of input samples.
    
    Parameters
    ------------
   * x,y: 1d ndarrays.
   * p: float,
       degree of variation.
   
   Returns
   -------
   * pvariation: 1d ndarray.
        
   Notes        
   -----------            
   * P-variation is calculated as:     
     f = sup(sum( d(x[k],x[k-1])^p )  )^(1/p).
   * See for more https://en.wikipedia.org/wiki/P-variation.
   * Special cases: 
     case p=1 is also called total variation.
     case p=2 is also called quadratic variation.
     if p=0, then it will be log(x) variation.
   * If y is not None, than variation of x-y will be taken.
        
    '''    
    x = np.asarray(x)
    if y is None: y = x
    else: 
        y = np.asarray(y)
        if (y.shape != x.shape): raise ValueError('y.shape ! = x.shape')    

    pp = np.zeros(x.shape,dtype = x.dtype)
    # TODO: need to be otimized, cycles work to slow!
    for i in range(x.shape[0]):
        for j in range(i): 
            pp[i] = np.max([pp[i],pp[j] + np.power(np.abs(x[i]-y[j]),p) ])    
    
    if(p !=0):        
        return np.power(pp[::-1],1/p)
    else:
        return pp[::-1]
             
# wiki p-variation
# double p_var(const std::vector<double>& X, double p) {
# 	if (X.size() == 0)
# 		return 0.0;
# 	std::vector<double> cum_p_var(X.size(), 0.0);   // cumulative p-variation
# 	for (size_t n = 1; n < X.size(); n++) {
# 		for (size_t k = 0; k < n; k++) {
# 			cum_p_var[n] = std::max(cum_p_var[n], cum_p_var[k] + std::pow(std::abs(X[n] - X[k]), p));
# 		}
# 	}
# 	return std::pow(cum_p_var.back(), 1./p);
# }
