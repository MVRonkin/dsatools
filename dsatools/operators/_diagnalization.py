import numpy as np
import scipy 

#--------------------------------------------------
def diaganal_average(matrix, reverse = True, 
                     samesize = False, averaging = True):
    '''
    Hankel averaging 
        (or diaganale averaging) of the matrix.

    Parameters
    ------------
    * matrix:  2d ndarray,
        is the input matrix.
    * reverse: bool,
        if True, backward diaganales 
        will be taken.
    * samesize: bool,
        if True, only diganal from the main 
        to the leftest will be taken.
    * averaging: bool,
        if True, mean value by each diaganale 
        will be taken, else summ insted of mean.
    
    Returns
    -----------
    * vector: 1d ndarray,
        if samesize = True
        with size = matrix.raws+matrix.columns-1
        if samesize = False
        size= (matrix.raws+matrix.columns-1)//2. 
    
    Notes
    ------------
    * If samesize = False:
      if reverse = False, 
         the diaganles from left bottom  
         to the right upper will be taken
      if reverse = True,  
         the diaganles from right bottom 
         to the left upper ([0,0] element) will be taken.
    * If samesize = True:
      if reverse = False, 
         the diaganles from left bottom 
         to the main one will be taken
      if reverse = True, 
         the diaganles from right bottom 
         to the main one will be taken.               

   Example
   --------------

    '''

    (raws, columns) = matrix.shape
    n_diags =  raws+columns-1
    
    if(samesize):
        n_diags = n_diags//2+1
        
    out = np.zeros(n_diags, dtype = matrix.dtype)
    
    for idx_from_bottom in np.arange(n_diags):
        idx = idx_from_bottom - raws + 1
        
        diag = get_diaganal(matrix, idx,reverse = reverse)

        if(not reverse):
            if averaging:
                out[idx_from_bottom] = np.mean(diag)
            else:
                out[idx_from_bottom] = np.sum(diag)
        else:
            if averaging:
                out[n_diags-idx_from_bottom-1] = np.mean(diag)
            else:
                out[idx_from_bottom] = np.sum(diag)
    return out

#--------------------------------- 
def get_diaganal(matrix, idx,reverse = False):
    '''
    Get idx diaganale of matrix, 
    counting from zero diag position.

    Parameters
    ------------
    * matrix:  2d ndarray,
        is the input matrix.
    * idx  int,
        is the index of diganale 
        from main diganale (zero diag).
    * reverse: bool,
        if True, backward diaganales 
        will be taken.
    
    Returns
    -----------
    * diag: 1d ndarray.
    
    Notes:
    -------------
    * if reverce = False: 
        idx = 0 - main  diaganale
        idx > 0 - left  direction
        idx < 0 - right direction  
    * if reverce = True:  
        idx = 0 - main  backward diaganale
        idx > 0 - right  direction
        idx < 0 - left direction                                

    Example
    -------------
    a = [1,2,3,4,5]
    b = signals.matrix.toeplitz(a)[:3,:]
    print(b)
    print(get_diaganal(b, 0))  # zero diaganale
    print(get_diaganal(b, -2)) # 2 diaganel in left direction
    print(get_diaganal(b, 3))  # 3 diaganel in right direction
    print(get_diaganal(b, 0,reverse=True))  # zero backward diaganale
    print(get_diaganal(b, -1,reverse=True)) # 1 right backward diaganale
    print(get_diaganal(b, 1,reverse=True))  # 1 left backward diaganale
            
    '''
    (raws, columns) = matrix.shape
    n_diags =  raws+columns-1
    
    idx_from_bottom = idx + raws-1
    
    if(idx_from_bottom>=n_diags or idx_from_bottom<0):
        raise ValueError('idx value out of matrix shape ')

    len_of_diag = _length_of_diag_(matrix, idx_from_bottom)

    out = np.zeros(len_of_diag, dtype = matrix.dtype)

    if(not reverse):
        if(idx>=0):   
            for i in np.arange (len_of_diag):
                out[i] = matrix[i,i+idx] 

        if(idx<0):  
            idx = np.abs(idx)
            for i in np.arange (len_of_diag):
                out[i] = matrix[i+idx,i] 

    else:
        if(idx>=0):   
            for i in np.arange (len_of_diag):
                indexes = columns-1 -i-idx
                out[i] = matrix[i,indexes] 

        if(idx<0):  
            idx = np.abs(idx)
            for i in np.arange (len_of_diag):
                idx = np.abs(idx)
                indexes = columns-1 -i
                out[i] = matrix[i+idx,indexes]
    return out

#--------------------------------------------------
def _length_of_diag_(matrix, idx):
    '''
    Get length of 
        idx diaganal of matrix.
    
    Parameters
    -----------
    * matrix:  2d ndarray,
        is the input matrix.
    * idx  int,
        is the index of diganale 
        from main diganale (zero diag).
    
    Returns
    ----------
    * len: int,
        length of diaganal.
    
    Notes
    ----------
    * idx is calculated from element 0,0,
        Thus, for isntance 0 diaganale has length 1
        the next one has length 2.

   Examples
   -----------
            
    '''    
    matrix = np.asarray(matrix)
    
    (raws, columns) = matrix.shape
    
    n_diags =  raws+columns-1
    
    if(idx>=n_diags):
        raise ValueError('index is out of diaganal number, ',n_diags)
    
    length = 0
    
    rank = min(raws, columns)
    
    if (idx<n_diags//2):
        length = min(idx+1,rank)
    else:
        length = min(n_diags-idx,rank)
    
    return max(length,0)