import numpy as np
import scipy


__all__ = ['_check_list','_rand_uniform']

#------------------------------------------------------------------------------------
def _check_list(values, required_length = 1, expander_value = 0):
    '''
     Check of input types and length in corresponding to requreid.
        If length of values (input) increas required one, it will be cut,
        if length less than it is required it can be extanded by expander_value.
     
     Paramteres
     ----------
     * values: 1d ndarray,
         input values.
     * required_length: int, 
         required lrngth, 
         if length of values (input) increas required one, 
         it will be cut,if length less than it is required 
         it can be extanded by expander_value. 
          If required_length<0, than this 
          requirements will not be apply.
     * expander_value: string,
         values for extend list of values if it necessary;
         special cases: 
         * 'last' - for extend by the last value of the inputs;
         * 'None' - extend by None.
     
     Returns
     ----------
     *: list ofrequired length.
                                    
    '''
    
    if(type(values) not in [list, tuple, np.ndarray]):
        values = np.array([values])
    
    if (required_length <0):
        return values
        
    if(len(values) > required_length):
        values[:required_length]
        
    elif(len(values) < required_length):
        
        if(expander_value is 'last'):
            expanders= values[-1]*np.ones(required_length- len(values))
            
        elif(expander_value in ['None', None]):
            expanders= np.array((required_length- len(values))*[None])
                                          
        else:
            expanders = expander_value*np.ones(required_length- len(values))

        values = np.hstack ((  values, expanders  ))   
    
    return values

#------------------------------------------------------------------------------------
def _rand_uniform( range_limits = [0,1], size =1, scale_x = 10):
    ''' 
    Create random uniformly dstributed values list.
    
    Parameters
    ------------
    * range_limits: [float,float]
        limits or bands of values.
    * size: int,
        size of output array;
    * scale_x: int
        scalue to transform int values to float operations.
    
    Returns
    ----------
    * return: list
        list of values.
        
    '''
#     range_limits  = [range_limits[0]*scale_x,range_limits[1]*scale_x]
    out = range_limits[0]*np.ones(size)

    if(range_limits[0]>range_limits[1]):
        range_limits = [range_limits[1], range_limits[0]]
     
    if(range_limits[0] != range_limits[1] ):  
        out = np.random.randint(0,scale_x,size)*(range_limits[1] - range_limits[0])/scale_x + range_limits[0]

    return list(out)