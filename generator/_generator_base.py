import numpy as np
import scipy

import sys

from .. import utilits as ut

from ._simsig_tools import _check_list,_rand_uniform

__all__ = ['generator_base']

from abc import ABC, abstractmethod
class generator_base(ABC):
    _signal_parameters = [] # dictionary list
    _system_parameters = {}
    _signal_parameters_dict_default = {}
    _system_parameters_dcit_default = {}
    
    #------------------------------------------------------------------------------------         
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.set_signal_parameters_dict_default()
    
    #------------------------------------------------------------------------------------         
    @abstractmethod
    def set_system_parameters(self, *args, **kwargs):
        pass
    
    #------------------------------------------------------------------------------------         
    #@prepared method
    def get_system_parameters(self):
        ''' 
        Get systemparameters. 
        '''
        return self._system_parameters
    
    #------------------------------------------------------------------------------------         
    #@prepared method   
    def set_signal_parameters_dict_default(self):
        ''' 
        Set dictionary list by default dictionary.
        '''  
        self.clear_signal_parameters()
        self.add_signal_parameters_dict_default()
    
    #------------------------------------------------------------------------------------         
    #@prepared method   
    def clear_signal_parameters(self):
        ''' 
        Clear dictionary list.
        '''         
        self._signal_parameters = []
    
    #------------------------------------------------------------------------------------         
    #@prepared method 
    def add_signal_parameters_dict_default(self):
        ''' 
        Add default dictionary to the dictionary list.
        '''         
        self._signal_parameters.append(self._signal_parameters_dict_default.copy()) 
    
    #------------------------------------------------------------------------------------             
    #@prepared method 
    def add_signal_parameters_dicts(self, dictionaries = None ):
        ''' 
        Add dictionaries to the dictionary list.

        Parameters
        -----------
        * dictionaries: dict, 
             dictionaries for list.
        
        Notes
        -----------
        * list, np.ndarray and tuple formats of input are applied.
        
        '''          
        if(dictionaries):            
            if(type(dictionaries) in [list, tuple, np.ndarray] ):
                
                #TODO: add check check format
                for dict_i in dictionaries:
                    self._signal_parameters.append( dict_i )           
        else:
            self.add_signal_parameters_dict_default()
    
    #------------------------------------------------------------------------------------         
    #@prepared method          
    def set_signal_parameters_dicts(self, dictionaries = None ):
        ''' 
        Set dictionary list.
        Notes: list, np.ndarray and tuple formats of input are applied.
        param dictionaries: - dictionaries for list.            
        '''
        self.clear_signal_parameters()
        self.add_signal_parameters_dicts(dictionaries)
    
    #------------------------------------------------------------------------------------         
    #@prepared method         
    def remove_signal_parameters(self, n_of_parameters_dict = -1):     
        ''' 
        Remove one dictionary form dictionary list.
        
        Parameters
        ------------
        n_of_parameters_dict: dict,
            number of the dict to remove.            
        '''
        if(n_of_parameters_dict>=len(self._signal_parameters)):raise(IndexError)        
        else:self._signal_parameters.pop(n_of_parameters_dict)
    
    #------------------------------------------------------------------------------------  
    @abstractmethod
    def make_signal_parameters_dict(self, *args, **kwargs):
        ''' 
        Make signal_parameters dictionary.   
        ''' 
        signal_parameters_dict = self.get_signal_parameters_dict_default()
        return signal_parameters_dict

    #------------------------------------------------------------------------------------         
    @abstractmethod
    def add_signal_parameters(self, *args, **kwargs):
        '''
        Add signal_parameters.            
        Notes
        --------
        * formats of the input: float, list, tuple, ndarray. 
        '''         
        pass
    #------------------------------------------------------------------------------------         
    @abstractmethod
    def set_signal_parameters(self, *args, **kwargs): 
        '''
        Set signal_parameters.            
        Notes
        --------
        * formats of the input: float, list, tuple, ndarray.   
        '''  
        pass
    
    #------------------------------------------------------------------------------------   
    @abstractmethod
    def add_random_signal_parameters(self, *args, **kwargs):  
        '''
        Add random uniformly distributed signal_parameters.            
        
        Notes
        -----------
        * callbacks doesnot applied for this function.      
        '''
        pass
    
    #------------------------------------------------------------------------------------ 
    #@prepared method
    def get_signal_parameters_dicts(self):
        ''' 
        Get signal parameters dictionary list. 
        '''
        return self._signal_parameters    

    #------------------------------------------------------------------------------------ 
    #@prepared method
    def print_signal_parameters(self ):
        ''' 
        Print signal parameters dictionary list. 
        '''
        print(30*'-')
        for i,dict_i in enumerate(self._signal_parameters): print(i,dict_i)

    #------------------------------------------------------------------------------------ 
    #@prepared method
    def get_signal_parameters_dict_default(self ):
        '''         
        Get signal parameters default dictionary. 
        '''
        return self._signal_parameters_dict_default.copy()  
     
    #------------------------------------------------------------------------------------ 
    #@prepared method    
    def _sim_signal(self, snr_db=None, signal_indexes = None):   
        ''' 
        Simulate signal in form corresponing the class. 
        
        Parameters
        ------------
        * snr_db: float,
            is the signal-to-noise ratio (in dB) for whole signal.
        * signal_indexes: int,
            index if signals in the signal dictionary.        
        
        Notes
        ---------
        * int,float,complex, list, np.ndarray and tuple 
           formats of signal_indexes are applied.
        * SNR in dB.
        * If callback is not None: 
          s = callback(s) (format of callback = f(x)),
          if callback is None it does not applied.

        '''
        params = np.array(self._signal_parameters)
        
        if signal_indexes is None:
            signal_indexes = np.arange(params.shape[0])
        
        elif (isinstance(signal_indexes,(int,float,complex))):
            signal_indexes = np.append([],signal_indexes)
        
        elif (type(signal_indexes) not in [list, tuple,np.ndarray]):
            raise ValueError(' applied types of signal_indexes are: int,')
               
        if (np.max(signal_indexes)>params.shape[0]):
            raise ValueError(' indexs range exeed length of parameters, ',params.shape[0])
            
        params = params[signal_indexes]
        
        for i,param_dict in enumerate(params):
            
            signew = self._sim_one_sig(param_dict)
            if(i==0):
                self._signal = signew
            else:
                self._signal = np.add(self._signal,signew )
        
        if(snr_db):    
            self._signal = ut.awgn(self._signal, snr_db, units = 'db')  
    
    #------------------------------------------------------------------------------------
    #@prepared method 
    def get_signal(self, snr_db=None): 
        '''
        Get signal in form corresponing to the class.
        
        Parameters
        ------------
        * snr_db: float,
            is the signal-to-noise ratio 
            (in dB) for whole signal.
        
        Returns
        -----------
        * signal: 1d ndarray. 
        
        Notes
        -----------

        '''        
        self._sim_signal(snr_db=snr_db, signal_indexes = None)
        return self._signal
    
    #------------------------------------------------------------------------------------
    def __call__(self, snr_db=None):
        '''
        Get signal in form corresponing to the class.
           
        Parameters
        ------------
        * snr_db: float,
            is the signal-to-noise ratio 
            (in dB) for whole signal.
        
        Returns
        -----------
        * signal: 1d ndarray. 
        
        Notes
        -----------
        
        '''
        return self.get_signal(snr_db)
    
    #------------------------------------------------------------------------------------
    @abstractmethod
    def _sim_one_sig(self, sig_param): 
        '''
        Simulate harmonic (actually fmcw signal).
        
        Parameters
        -----------
        * sig_param: dict, 
            dictionary of signal parameters.
          
        Returns
        ----------
        * signal:1d naarray,
            simulated signal.
        '''
        sig = np.zeros([],dtype = np.complex)
        
        if (callback in  ['None', None]):
              return sig

        elif type(callback ) is not list:
            callback = list([callback])

            for callback_i in callback:
                sig = callback_i(sig)
           
            return sig   