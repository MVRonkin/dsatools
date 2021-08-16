import numpy as np
import scipy
import matplotlib.pyplot as plt

from ._math_auxilary import afft

#-------------------------------------------------------------------
def probe(x, figsize = (12,4), title = None, save_path = None,  plt_settings = None):
    '''
    Probe plot of input signal,
        plot its real-values in time domain
        and its amplitude spectrum.
    
    Parameters
    ----------
    * x: 1d ndarray.
    * figsize: tuple(int,int),
        size of figure.
    * title: string,
        figure title.
    * save_path: string or None,
        if not None, fig will be saved.
    * plt_settings: *args, **kwargs or None,
        additional matplotlib.pyplot settings.
        
    '''
    plt.figure(figsize = figsize)
    
    if(plt_settings):
        plt.subplot(121)
        plt.plot(np.real(x),plt_settings)
        plt.subplot(122)
        plt.plot(afft(x),plt_settings)
    else:
        plt.subplot(121)
        plt.plot(np.real(x),'k')
        plt.subplot(122)
        plt.plot(afft(x),'k')
        
    if title != None:
        plt.title(label=title)
        
    if save_path != None:
        plt.savefig(save_path) 
    plt.show()    
#     if(finish):
#         plt.show()

#-------------------------------------------------------------------   
def probe_filter(filterwindow, figsize = (12,4), title = None, 
                 save_and_path = False, finish = True, plt_settings = None):
    
    fig, ax = plt.subplots(2, 1, figsize=figsize)

    if(plt_settings):
        ax[0].plot(20*np.log10(abs(filterwindow)), plt_settings)
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='black')
        ax[0].grid()
        
        ax[1].plot(np.unwrap(np.angle(filterwindow))*180/np.pi,plt_settings)
        ax[1].set_ylabel("Angle (degrees)", color='black')
        ax[1].set_xlabel("Frequency (points)")
        ax[1].grid()
    else:
        ax[0].plot(20*np.log10(abs(filterwindow)), color='blue')
        ax[0].set_title("Frequency Response")
        ax[0].set_ylabel("Amplitude (dB)", color='black')
        ax[0].grid()
        
        ax[1].plot(np.unwrap(np.angle(filterwindow))*180/np.pi, color='green')
        ax[1].set_ylabel("Angle (degrees)", color='black')
        ax[1].set_xlabel("Frequency (points)")
        ax[1].grid()
        
    if title != None:
        plt.title(label=title)
        
    if save_and_path != False:
        plt.savefig(save_and_path) 
        
    if(finish):
        plt.show()   
