"""
The aim of this script is to reproduce data from Dhamala et al. (2008) article.

The two processes can be simulating using either AR_Dhamala_1 or AR_Dhamala_2 (see AR_func variable)
and anlyzed either with a basic 'fft' method
or a wavelet-based method but currently using the OpenElectrophy implementation (to use : install the OpenElectrophy python module)

"""

from scipy import *
from scipy.fftpack import fft,fftfreq
import matplotlib.pyplot as py 
from itertools import combinations_with_replacement,combinations
from NPSpectralGranger import NPSpectralGranger
import time

try:
    import neo
    from connection import * # To import OpenElectrophy, more standard command : from OpenElectrophy import *
    import quantities as pq
except:
    print "OpenElectrophy is not installed, the 'wavelet' method will not be available"


def AR_Dhamala_1(N,stds=ones(3)):
    """
    Construct a AR process from Dhamala model 1 of length N
    In this process : Y -> Z -> X 
    """

    out=zeros((3,N))
    X,Y,Z=out[0,:],out[1,:],out[2,:]
    
    for i in range(2,N):
        X[i]=0.8*X[i-1]-0.5*X[i-2]+0.4*Z[i-1]+stds[0]*randn()
        Y[i]=0.53*Y[i-1]-0.8*Y[i-2]+stds[1]*randn()
        Z[i]=0.5*Z[i-1]-0.2*Z[i-2]+0.5*Y[i-1]+stds[2]*randn()
        
    return out

def AR_Dhamala_2(N,stds=ones(2)):
    """
    Construct a AR process from Dhamala model 2 of length N,
    change of coupling direction occurs at the mid-point
    """

    out=zeros((2,N))
    Y1,Y2=out[0,:],out[1,:]
    
    int_1=0.5/(1+exp(-(arange(N)-N//2)/100.))
    int_2=0.5-int_1
    for i in range(2,N):
        Y1[i]=0.53*Y1[i-1]-0.8*Y1[i-2]+int_1[i]*Y2[i-1]+stds[0]*randn()
        Y2[i]=0.53*Y2[i-1]-0.8*Y2[i-2]+int_2[i]*Y1[i-1]+stds[1]*randn()
        
    return out

plot_all=True

n_trials=100
n_points=4000
method='fft' # 'fft' or 'wavelet'
ARfunc=AR_Dhamala_1 # AR_Dhamala_1 or AR_Dhamala_2
stds=[1.,0.5,0.5]

names=['X','Y','Z']

if ARfunc==AR_Dhamala_1:
    nsig=3
elif ARfunc==AR_Dhamala_2:
    nsig=2
    
if method=='fft':
    spectrums=zeros((n_points,nsig,nsig),dtype='complex128') # For standard fft
elif method=='wavelet':
    f_start=0.*pq.Hz
    f_stop=100.*pq.Hz
    deltafreq=1.*pq.Hz
    n_freqs=arange(f_start,f_stop,deltafreq).size
    spectrums=zeros((n_freqs,nsig,nsig,n_points),dtype='complex128') # For wavelets
else:
    print "Non valid method"
    exit(1)
    
import time
t0=time.time()
print "Start AR computing"
for nt in range(n_trials):
    if mod(nt,100)==0:
        print "Trials : ",nt
    XYZ=ARfunc(n_points,stds=stds)

    # Compute all cross spectral densities
    if method=='fft':
        # Idea 1 : simply compute fft over trials
        XYZfft=fft(XYZ,axis=1)
        # Add to spectrums
        for i,j in combinations_with_replacement(range(3),2):
            spectrums[:,i,j]+=XYZfft[i,:]*XYZfft[j,:].conj() # Spectrums of each trial are directly summed
            
        if 0: # To plot the signals and their FFT spectrums
            py.figure()
            X,Y,Z=XYZ[0,:],XYZ[1,:],XYZ[2,:]
            py.plot(X)
            py.plot(Y)
            py.plot(Z)

            py.figure()
            freqs=fftfreq(n_points,1./200.)
            Xfft,Yfft,Zfft=XYZfft[0,:],XYZfft[1,:],XYZfft[2,:]
            py.plot(freqs,abs(Xfft))
            py.plot(freqs,abs(Yfft))
            py.plot(freqs,abs(Zfft))

    elif method=='wavelet':
        # Idea 2 : compute wavelet spectrums
        anas,tfs=[],[]
        for i in range(XYZ.shape[0]):
            anas.append(neo.core.AnalogSignal(XYZ[i,:],sampling_rate=200.*pq.Hz,units=pq.V,))            
            tfs.append(TimeFreq(anas[i],
                                    f_start=f_start,
                                      f_stop=f_stop,
                                      deltafreq=deltafreq,
                                      sampling_rate=200.,
                                      optimize_fft=False,
                                ))
        # Add to spectrums
        for i,j in combinations_with_replacement(range(XYZ.shape[0]),2):
            for k in range(n_points):
                spectrums[:,i,j,k]+=tfs[i].map[k,:]*tfs[j].map[k,:].conj()

# Average spectrums and make it symetric
if method=='fft':
    #~ spectrums=spectrums.sum(axis=3)
    print "Time for AR generation and spectrum computation : ",time.time()-t0
    spectrums/=n_trials
    for i,j in combinations(range(spectrums.shape[1]),2):
        spectrums[:,j,i]=spectrums[:,i,j].conj()

    freqs=fftfreq(n_points,1./200.)
    if plot_all:
        # Plot spectrums
        fig=py.figure()
        fig.suptitle("Double-sided spectrums computed with FFT")
        for i,j in combinations_with_replacement(range(spectrums.shape[1]),2):
            py.plot(freqs,abs(spectrums[:,i,j]))

    freq_end=n_points//2+1 # Nyquist must be included
    
    fig=py.figure()
    fig.suptitle("Read as 'does column influence line ?'")
    ax=empty((3,3),dtype='object')
    for n1 in range(3):
        for n2 in range(3):
            if n1!=n2:
                ax[n1,n2]=fig.add_subplot(3,3,7+n2-3*n1)
                if n1==0:
                    ax[n1,n2].set_xlabel(names[n2])
                if n2==0:
                    ax[n1,n2].set_ylabel(names[n1])
                    
    for mode in ['bivariate','multivariate','conditional']:
        print "Start granger, mode : ",mode
        t1=time.time()
        NPGr=NPSpectralGranger(spectrums,abs(freqs[:freq_end]),mode=mode,Niterations=20)
        print "Done in ",time.time()-t1
        for n1 in range(3):
            for n2 in range(3):
                if n1!=n2:
                    ax[n1,n2].plot(abs(freqs[:freq_end]),NPGr[n1,n2])

elif method=='wavelet':

    freqs=arange(f_start,f_stop,deltafreq)

    # Make spectrums symetric
    spectrums=concatenate([spectrums,spectrums[:0:-1,:,:,:].conj()],axis=0)
    for i,j in combinations_with_replacement(range(spectrums.shape[1]),2):
        spectrums[:,j,i,:]=spectrums[:,i,j,:].conj()
    assert(spectrums.shape==(2*freqs.size-1,spectrums.shape[1],spectrums.shape[1],n_points))

    
    print "Start computing all bivaraite Granger over time"
    for pair in combinations(range(spectrums.shape[1]),2):
        print "for pair : ",pair
        print
        time_Granger_1=[]
        time_Granger_2=[]
        for np in range(0,n_points,20):
            inds=ix_(range(spectrums.shape[0]),pair,pair,array([np]))
            granger=NPSpectralGranger(spectrums[inds].squeeze(), freqs,mode='bivariate',Niterations=20)
            time_Granger_1.append(granger[0,1])
            time_Granger_2.append(granger[1,0])
        print
        
        fig=py.figure()
        ax=fig.add_subplot(211)
        ax.imshow(array(time_Granger_1).transpose())
        ax=fig.add_subplot(212)
        ax.imshow(array(time_Granger_2).transpose())
        fig.suptitle("%s and %s"%(str(names[pair[0]]),str(names[pair[1]])))
            
                
py.show()
        
        