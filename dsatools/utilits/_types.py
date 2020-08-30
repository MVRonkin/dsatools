import numpy as np
import scipy


__all__=['fixpoint','is_1d','is_complex','to_1d','to_2d','to_callback',]
#-------------------------------------------------------------------
# def is_numbertype(x): 
#     return isinstance(x,(float,int,list,tuple,complex,np.ndarray,np.matrix))

#-------------------------------------------------------------------
tupleCorrection = lambda x: np.append([],x)[0]

def fixpoint(x,d=2):
    '''
    For print with restricted length.
    '''
    return np.fix(x*10**d)/10**d

#-------------------------------------------------------------------
def is_1d(x):    
    if(np.ndim(x)!=1): return False    
    else: return True

#-------------------------------------------------------------------
def is_complex(x):
    x = np.asarray(x)
    return isinstance(x.dtype,complex.np._complex,np.complex,np.complex128,np.complex64) 
#-------------------------------------------------------------------
def to_1d(x):
    x = np.asarray(x)
    if (np.ndim(x)==0): return np.append([],x)
    else:  return x
#-------------------------------------------------------------------    
def to_2d(x, column=False):
    x = np.asarray(x)
    x=to_1d(x)
    
    if(len(x.shape) == 1): 
        if column:
            return x[np.newaxis,:]
        else:
            return x[:,np.newaxis]
    else: 
        return x

#-------------------------------------------------------------------
class to_callback:
    def __init__(self, function,n_args2call=2,n_kwargs2call=0, *args, **kwargs):
        self.function = function
        self.args     = args
        self.kwargs   = kwargs
        self.n_args2call   = n_args2call
        self.n_kwargs2call = n_kwargs2call
    #-------------------------------------------------------------------    
    def __call__(self, *args2call,**kwargs2call):        
        return self.function(*args2call[:self.n_args2call],*self.args,**kwargs2call,**self.kwargs ) 
    
#-------------------------------------------------------------------
class to_callback_xy(to_callback): 
    def __init__(self, function,*args, **kwargs):
        super().__init__(function, 2, 0,*args, **kwargs )
        
#-------------------------------------------------------------------        
class to_callback_x(to_callback): 
    def __init__(self, function,*args, **kwargs):
        super().__init__(function, 1, 0,*args, **kwargs ) 
        
