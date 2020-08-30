import numpy as np

__all__ = ['tau_fullphase','maxcor','maxcor_real']

#-------------------------------------------------------------- 
def tau_fullphase(s1,s2=None,f0=1,delta_f=1,Tm=None,fs=None,w_on=True):   
    '''
    Time delay difference estimations
     between two complex-valued  beat signals, obtained by frequency 
     modulated continium waves signals (FMCW). Method is based on the 
     phase-to-time approximation of the beat signals by the weigthed 
     least-square method, with weigths equal to its modules.
          
    Parameters
    ----------------
    * s1: 1d ndarray (complex),
        is the input signal.
    * s2: 1d ndarray (complex),
        if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated. 
    * f0: float,
        is the initial frequency.
    * delta_f: float,
        is the frequency band.    
    * T_m: float,
        is the period of modulation
        (T_m = x.shape[0]/fs if None).
    * fs: float,
        sampling frequency 
        (if None fs = x.shape[0]).

    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.

    Notes
    --------------------------
    * If fs=x.shape[0], than delay will be calculated in points units.
    * The method is based on the supposition, that full phase 
        (frequency*time+inital_phase) is depends only on delay 
        (or delay difference).
    * The value of time dealys has restricted 
        unambiguous estimation range +-1/2\\pi(f_0+\\Delta f/2)
    * If s2 is None, than estimation will be perfermed for dealy of s1.
    * Basic estimator:
      ..math::
      Delta_tau =sum_n{W[n]|s[n]|angle(s[n])}/sum{W^2[n]|s[n]|},
      where:
      * Delta_tau is the estimated time delay difference;
      * s[n] = s1[n]*conj(s2[n]);
      * s1,s2 - beat signals time delay difference beyween 
        which is measured;
      * angle(s) is the angle (argument) of complex-valued signals;
      * W[n] = 2pi((Delta f*T_m)*(n*fs) + f_0),
        where:
        * f_0 is the initial frequency of FMCW signal;
        * Delta f is the frequency band (frequency deviation) of 
           the corresponing FMCW signal (from f_0 to f_0+Delta_f);
        * T_m is the period of modulation;

    * The estimator is based on the following beat signal model:
      ..math::
      s[n] = a[n](exp{2j\\piW[n]\\tau})+s_par[n]+noises[n], 
        a[n]>|s_par[n]|  (high signal-to-interferences ratio),
        a[n]>|noises[n]| (high signal-to-noises ratio),
        where:
        * a[n] is the amplitude of valuable signal;
        * s_par[n] are the influence of the interference signals;
        * noises are the white gaussian noies.

    References
    -------------------
    [1] Ronkin M.V., Kalmykov A.A., Zeyde K.M. 
        Novel FMCW-Interferometry Method Testing on an 
        Ultrasonic Clamp-on Flowmeter, IEEE Sensors Journal,  
        Vol 20 , Issue 11 , 2020  p. 6029 - 6037,  
        DOI: 10.1109/JSEN.2020.2972604.
    [2] Ronkin M.V., Kalmykov A.A. Investigation of the time 
        delay difference estimator for FMCW signals, 
        Proceedings of the 2nd International Workshop on 
        Radio Electronics & Information Technologies 
        (REIT 2 2017), 2017. p. 90-99, 
        http://ceur-ws.org/Vol-2005/paper-11.pdf.  
        
    '''    
    s, N = __check_input__(s1,s2)
    if(fs is None): fs = N        
    if(Tm is None): Tm = N/fs    

    Kth  = 2*np.pi*f0
    Kf   = delta_f/Tm
    Kw   = Kf*2*np.pi/fs
    n    = np.arange(N)    
    W    = n*Kw+Kth # W = 2*np.pi *(delta_f*n/(Tm*fs) + f0) 
      
    if (w_on==True):
        t_est = _direct_tau_est(s,W,np.abs(s))
        
    else:
        t_est = _direct_tau_est(s,W,1)
 
    return   t_est

#--------------------------------------------------------------  
def maxcor(s1,s2=None,f0=1,delta_f=0):
    '''
    Time delay difference estimations
     between two continium complex-valued signals, 
     including beat signal, obtained by the frequency modulated 
     continium waves signals (FMCW), and FMCW signals its self.
   
    Parameters
    ----------------
    * s1: 1d ndarray (complex),
        is the input signal.
    * s2: 1d ndarray (complex),
        if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated. 
    * f0: float,
        is the initial frequency.
    * delta_f: float,
        is the frequency band.    
    
    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.       

    Notes
    -----------------------
    * method does not require frequrency deviation, if delta_f =0,
        than method gives dealy estimation by initial phase.        
    * The value of time dealys has restricted 
              unambiguous estimation range +-1/2\\pi(f_0+\\Delta f/2).
    * If s2 is None, than estimation will be perfermed for dealy of s1.
    * The estimator expression
      ..math::
      Delta_tau = arccos(sum_n{rho0[n])/2pi(f_0+Delta f/2),
      where:
      * Delta_tau is the estimated time delay difference;
      * rho0[n] = s1[n]*conj(s2[n])/sqrt(sum(s1^2*s2^2));
      * s1,s2 - beat signals time delay difference beyween 
          which is measured;
      * angle(s) is the angle (argument) of the complex value;
      * f_0 is the initial frequency of FMCW signal;
      * Delta_f is the frequency band (frequency deviation) of 
        the corresponing FMCW signal (from f_0 to f_0+Delta_f);
      * T_m is the period of modulation.

    References
    ----------------------
    [1] Liao Y, Zhao B.,
        Phase-shift correlation method fot accurate phase difference
        estimation in range fider, Application optic, 
        v.54 # 11 p. 3470-3477.
    [2] Bjorklund S.,A survey and comparison of time-delay estimation 
        methods in linear systems.— UniTryck: Linkoping, Sweden, 
        2003. —169 p.
    '''
    s, N = __check_input__(s1,s2)
    corcof = np.sum(s)
    angle  = np.angle(corcof)
    return  angle/(2*np.pi*f0+np.pi*delta_f) 


#--------------------------------------------------------------  
def maxcor_real(s1,s2=None,f0=1,delta_f=0):
    '''
    Time delay difference estimations
     between two continium real-valued signals, 
     including beat signal, obtained by the frequency 
     modulated continium waves signals (FMCW), 
     and FMCW signals its self.

    Parameters
    ----------------
    * s1: 1d ndarray (complex),
        is the input signal.
    * s2: 1d ndarray (complex),
        if s2 is not none, than the dealy of 
        the conjugated product of s1,s2 will be estmated. 
    * f0: float,
        is the initial frequency.
    * delta_f: float,
        is the frequency band.    
    
    Returns
    ------------------
    * delta_tau: float,
        estimated time delay difference.   

    Notes
    -----------------------
    * method does not require frequrency deviation, if delta_f =0,
        than method gives dealy estimation by initial phase.        
    * The value of time dealys has restricted 
              unambiguous estimation range +-1/2pi(f_0+Delta_f/2).
    * If s2 is None, than estimation will be perfermed for dealy of s1.
    * The estimator expression
      ..math::
      Delta_tau = arccos(sum_n{rho0[n])/2pi(f_0+Delta_f/2),
      where:
      * \\Delta\\tau is the estimated time delay difference;
      * rho0[n] = s1[n]*conj(s2[n])/sqrt(\\sum(s1^2*s2^2));
      * s1,s2 - beat signals time delay difference beyween 
          which is measured;
      * angle(s) is the angle (argument) of the complex value;
      * f_0 is the initial frequency of FMCW signal;
      * \\Delta f is the frequency band (frequency deviation) of 
          the corresponing FMCW signal (from f_0 to f_0+Delta_f);
      * T_m is the period of modulation.

    References
    ----------------------
    [1] Liao Y, Zhao B.,
        Phase-shift correlation method fot accurate phase difference
        estimation in range fider, Application optic, 
        v.54 # 11 p. 3470-3477.
    [2] Bjorklund S.,A survey and comparison of time-delay estimation 
        methods in linear systems.— UniTryck: Linkoping, Sweden, 
        2003. —169 p.
    '''
    s1 = np.asarray(s1,dtype = np.float)
    if (s2 is not None): s2 = np.asarray(s2,dtype = np.float)
    s, N = __check_input__(s1,s2)

    corcof = np.sum(s)/np.sqrt(np.sum(np.square(s1))*np.sum(np.square(s2)))
    angle  = np.arccos(corcof)
    return  angle/(2*np.pi*f0+np.pi*delta_f) 

#--------------------------------------------------------------    
def _direct_tau_est(s,W,absS):
    ''' 
    Auxilary function
    '''
    ph     = (np.angle(s))
    Wnum   = np.sum(W*absS*ph)
    Wdenum = np.sum(absS*(np.square(W)))
    t_est  = Wnum/Wdenum
    return   t_est
#-------------------------------------------------------------- 
def __check_input__(s1,s2):
    s1 = np.array(s1)
    N = s1.shape[0]
    
    if(s2 is None):
        return s1, N
    
    s2 = np.array(s2)
    
    if(s1.shape != s2.shape):
        raise ValueError('s1.shape != s2.shape')
    
    s = s2*np.conj(s1)   

    return s, N
#--------------------------------------------------------------   

# def MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm): 
#     N   = len(s1)
#     Kth = 2*np.pi*F0
#     Kf  = incF/Tm    
#     Rf  = correlation.crossCorr_clear(s1,s2)
#     Rb  = correlation.crossCorr_clear(s2,s1)
#     R1  = correlation.crossCorr_clear(s1,s1)
#     R2  = correlation.crossCorr_clear(s2,s2)
    
    
#     Kf  = incF/Tm
#     Kth = 2*np.pi*F0            
#     n   = np.arange(N)
#     Kw  = Kf*2*np.pi/fs
    
#     W      = ((N-1+n)*Kw/2+Kth)     
#     t_est1 = DirCorrEst(Rf,R1,W)
#     t_est4 = DirCorrEst(Rb,R2,-W) 
#     W      = ((N-1-n)*Kw/2+Kth) 
#     t_est2 = DirCorrEst(Rf,R2,W) 
#     W      = ((N-1-n)*Kw/2+Kth) 
#     t_est3 = DirCorrEst(Rb,R1,-W)
#     W      = ((N-1)*Kw+2*Kth)
#     t_est5 = -DirCorrEst(Rb,Rf,W)
#     W      = ((N-1+n)*Kw+2*Kth)
#     t_est6 = -DirCorrEst(Rb*np.conj(R2),Rf*np.conj(R1),W)
#     W      = ((N-1-n)*Kw+2*Kth)
#     t_est7 = -DirCorrEst(Rb*np.conj(R1),Rf*np.conj(R2),W)
#     return   t_est1,t_est2,t_est3,t_est4,t_est5,t_est6,t_est7
# #--------------------------------------------------------------
# def fullPhiEst(s1,s2,fs,F0,incF,Tm):     
#     N   = len(s1)
#     Kth  = 2*np.pi*F0
#     Kf   = incF/Tm
#     Kw   = Kf*2*np.pi/fs
#     n    = np.arange(N)    
#     W    = n*Kw+Kth
    
#     s      = s1*np.conj(s2)
#     t_est1 = FMCWD_est(s,W,np.abs(s))
#     t_est2 = FMCWD_est(s,W,1)
    
#     s      = s1*np.conj(s2)
#     t_est3 = FMCWD_est(s,W,np.abs(s))
#     t_est4 = FMCWD_est(s,W,1)    
#     return   t_est1,t_est2,t_est3,t_est4



# #--------------------------------------------------------------  
# def FB_est(s1,s2,fs,F0,incF, Tm):
#     t_est = np.zeros(15)
    
#     t_est[0],t_est[1],t_est[2],t_est[3] = MyFullPhiEst(s1,s2,fs,F0,incF, Tm)
#     t_est[4],t_est[5],t_est[6],t_est[7],t_est[8],t_est[9],t_est[10] = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#     t_est[11] = NLphiComplexEst(s1,s2,F0,incF)
#     t_est[12] = NLphiRealEst(np.real(s1),np.real(s2),F0,incF)
#     t_est[13] = WLSphiEst(s1,s2,F0)
#     t_est[14] = LSphiEst((s1),(s2),F0)    
#     return t_est


# #--------------------------------------------------------------  
# def tautau_est(s, incF, f0, fs, Tm, max_tau_scale = 12):

#     '''
#     t_est[0],t_est[1],t_est[2],t_est[3] = MyFullPhiEst(s1,s2,fs,F0,incF, Tm)
#     t_est[4],t_est[5],t_est[6],t_est[7],t_est[8],t_est[9],t_est[10] = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#     t_est[11] = NLphiComplexEst(s1,s2,F0,incF)
#     t_est[12] = NLphiRealEst(np.real(s1),np.real(s2),F0,incF)
#     t_est[13] = WLSphiEst(s1,s2,F0)
#     t_est[14] = LSphiEst((s1),(s2),F0)
#     t_est[15] = t_coarse
#     '''
    
#     s = s - np.mean(s)    
    
#     max_tau  = 1/(max_tau_scale*(f0+incF)/2)

#     f1       = fitzR(s,fs)

#     t_coarse = f1*Tm/incF
    
#     t_int    = np.fix(t_coarse/max_tau)*max_tau
    
#     s_ref    = ut.sim_sig(t_int*incF/Tm, fs, len(s), 2*np.pi*f0*t_int)
    
# #    s_ref    = np.abs(s)*s_ref
    
#     tst      = FB_est(s, s_ref, fs, f0, incF, Tm) 
    
# #    tst      = tst *75/76 + 1/76 * t_coarse
    
#     tst      = tst +t_int
# #    t_expr = expirementPhiEst(s,s_ref,fs,f0,incF, Tm, N_shuffle = 40, M_shuffle = 40)
    
# #    tst = np.append(tst, t_coarse)
#     tst = np.append(tst, t_coarse)

    
    
#     return tst
# #-------------------------------------------------------------- 
# def tauPhase_est(s1,s2,fs,F0,incF,Tm, mod = 'NLR'):
#     '''
#     MODS = NLR, NLC, WLS, LS, TEST0-3, TEST4,5,6,7, TEST8, TEST9, TEST10, FITZR
#     '''
#     out = 0
#     if  (mod == 'NLR'):
#         out = NLphiRealEst(np.real(s1),np.real(s2),F0,incF)
#     elif (mod == 'NLC'):
#         out = NLphiComplexEst(s1,s2,F0,incF)
#     elif (mod == 'WLS'):
#         out = WLSphiEst(s1,s2,F0)
#     elif (mod == 'LS'): 
#         out = LSphiEst((s1),(s2),F0) 
#     elif (mod == 'test0'):
#         out,_,_,_ = MyFullPhiEst(s1,s2,fs,F0,incF, Tm)
#     elif (mod == 'test1'):
#         _,out,_,_ = MyFullPhiEst(s1,s2,fs,F0,incF, Tm) 
#     elif (mod == 'test2'):
#         _,_,out,_ = MyFullPhiEst(s1,s2,fs,F0,incF, Tm)        
#     elif (mod == 'test3'):
#         _,_,_,out = MyFullPhiEst(s1,s2,fs,F0,incF, Tm)
#     elif(mod == 'test4'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[0]
#     elif(mod == 'test5'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[1]  
#     elif(mod == 'test6'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[2] 
#     elif(mod == 'test7'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[3]
#     elif(mod == 'test8'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[4]
#     elif(mod == 'test9'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[5]
#     elif(mod == 'test10'):
#         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = outs[6]
#     elif(mod == 'FitzR' or mod == 'FITZR'):
# #         outs = MyFBPhiFullEst(s1,s2,fs,F0,incF, Tm)
#         out = FitzR3_est(s1,fs) - FitzR3_est(s2,fs) 
#     return out

# #--------------------------------------------------------------  
# def FB_est_light(s1,s2,fs,F0,incF, Tm):
#     N   = len(s1)
#     Kth = 2*np.pi*F0
#     Kf  = incF/Tm    
#     Rf  = crossCorr_clear(s1,s2)
# #     Rb  = crossCorr_clear(s2,s1)
#     R1  = crossCorr_clear(s1,s1)
# #     R2  = crossCorr_clear(s2,s2)
#     Kf  = incF/Tm
#     Kth = 2*np.pi*F0            
#     n   = np.arange(N)
#     Kw  = Kf*2*np.pi/fs
    
#     W      = ((N-1+n)*Kw/2+Kth)  
    
#     t_est1 = DirCorrEst(Rf,R1,W)
    
#     return t_est1
# #--------------------------------------------------------------  
# def tautau_est_light(s, incF, f0, fs, Tm, max_tau_scale = 12):
    
#     s = s - np.mean(s)    
    
#     max_tau  = 1/(max_tau_scale*(f0+incF)/2)

#     f1       = fitzR(s,fs)

#     t_coarse = f1*Tm/incF
    
#     t_int    = np.fix(t_coarse/max_tau)*max_tau
    
#     s_ref    = ut.sim_sig(t_int*incF/Tm, fs, len(s), 2*np.pi*f0*t_int)
    
#     if np.isreal(s[0]):
#         t_est = NLphiRealEst(np.real(s),np.real(s_ref),F0,incF)
#     else:
#         t_est = NLphiComplexEst(s,s_ref,F0,incF)    

#     tst      = tst +t_int
# #    t_expr = expirementPhiEst(s,s_ref,fs,f0,incF, Tm, N_shuffle = 40, M_shuffle = 40)

#     tst = np.append(tst, t_coarse)
    
#     return tst

# #-------------------------------------------------------------- 
# def ml_est(s,fs = 1,Nfft = None):
#     if(not Nfft):
#         Nfft    = len(s)
        
#     LimOfFind = int(Nfft/2)
#     S      = np.abs(np.fft.fft(s,Nfft))  # расчет квадрата амплитуды спектра
#     S      = np.hstack((np.zeros(1),S[1:LimOfFind]))
#     pp     = np.flatnonzero(S==max(S))   # поиск первого максимума 
#     f_res  = fs*(pp)/Nfft      # расчет частоты первого максимума  
#     return   f_res[0]