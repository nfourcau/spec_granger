import numpy as np
import scipy.io as scio
from matop import *
from numpy.linalg import LinAlgError

def Granger(S, freqs):
    """
    compute the granger causality for two neurons
    
    param:
        S: 3D auto/cross spectral density matrix with chanxchanxfreq we focus on [[S11, S12],[S21,S22]] per frequency
        freqs: frequenties at which the auto/cross spectral densities were computed
        
    """
    H, Z, S, psi = _FactorWilson(S, freqs)
    
    for directedPair, freqs, granger in _ComputeGranger(H, Z, S, psi, freqs):
        yield directedPair, freqs, granger, S, H, Z

def _ComputeGranger(H, Z, S, psi, freqs):
    """
    recieves the noise covariance matrix: Z, the cross-spectral density: S (1-sided)
    the transfer function H, and the left-spectral factor psi and computes the directed granger causality
    for two orientations of the given pair
    param: 
        Z: noise covariance matrix 2x2xfreq: [[Z11,Z12],[Z21,Z22]]
        H: transfer function 2x2xfreq: [[H11, H12],[H21,H22]]
        S: spectral densities [[S11, S12],[S21, S22]]
        psi: spectral factors
    yields:
        directedPair: pair (chanA, chanB) with the direction defined from chanA -> chanB 
        freqs: frequencies at which the granger is evaluated
        granger: granger causality
    
    """
    for pair in ((0,1), (1,0)): # flip the direction of the granger to get the other direction
        granger = np.log(S[pair[0], pair[0],:] /
        (S[pair[0], pair[0],:]-((Z[pair[1],pair[1]]-((Z[pair[0], pair[1]]**2) / Z[pair[0],pair[0]]))*(np.abs(H[pair[0],pair[1],:])**2))))
        
        yield pair, freqs, granger
    
def _FactorWilson(S, freqs, Niterations = 100, tol = 1e-19, init = 'chol'):
    """
    based on the sfactorwilson2x2.m from the feartrip toolbox
    matlab param:
        cmbindx: is a 2 column matrix with all pairwise combinations of channels under investigation
        tol, init: have to with the taming and initialization of the algorithm
    
    param:
        S: 3D matrix chanxchanxfreq with one-sided spectral density
            in our case this is a 2x2 matrix [[S11, S12],[S21, S22]]
        fs: (sampling frequency in Hz) isn't used here I think
        freqs: vector of frequencies at which the spectral densities are given  
        Niterations: number of repeats 
    output:
        For now the outputs take the shape of 4 rows for each combination H11;H12;H21;H22 etc
        H: (transfer function)
        Z: (noise covariance)
        S: (cross-spectral density 1-sided)
        psi: (left spectral factor)
        
        usage in feartrip, they get the auto/cross spectra with ft_freqanalysis
        Here they are putting multiple timeslices through the algorithm kkkkkkk
        [Htmp, Ztmp, Stmp] = sfactorization_wilson2x2(freq.crsspctrm(:,:,:,kk), ...
                      freq.freq, numiteration, tol, cmbindx, fb, init);
     
    """
    m = 1                           # only one pair for now
    N = np.size(freqs)-1            # why minus 1 (originally np.size(freqs)-1
    N2 = (2*N)                      # (originally 2*N with N=np.size(freqs)-1) for the Sarr matrix N2 is most likely 1 to low
    assert(N2 == (2*np.size(freqs))-2)  # check of frequency element size
            
    """
    Sarr has the shape 2*2*aCombinations*freqs
    - freqs is the double sided power spectrum
    
    """
    # Step 1: forming 2-sided spectral densities for ifft routine in matlab
    Sarr = S.copy() # data already is double sided and the right orientation (I assume)
    assert(np.size(freqs)-1==N) # see if N is consistent      
    assert(Sarr.shape == (2,2,N2))

    # Step 2: computing covariance matrices
    gam = np.real(np.fft.ifft(Sarr, axis = 2)) # ifft directly over the last axis
    assert(gam.shape == (2, 2, N2))
     
    gam0 = gam[:,:,0].copy() # auto/cross spectral densities of the lowest frequency
    assert(gam0.shape == (2,2))
    
    h = np.zeros(gam0.shape, dtype = np.complex) # 2x2 matrix used to store initial condition for optimization 
    if init == 'chol': # based on the cholesky decompositon
        try: # see if decompisition worked
            tmp = np.transpose(np.linalg.cholesky(gam0)) # numpy cholesky returns lower triangle, so transpose

        except LinAlgError: # if decomposition failed do random initialization
            tmp = np.random.rand(2,2)
            tmp = np.triu(tmp)

    elif init == 'rand': # random initial conditions
        tmp = np.random.rand(2,2)
        tmp = np.triu(tmp)

    else:
        assert(init not in ['chol', 'rand'])
        raise ArgumentError("initialization option either 'chol' or 'rand'")
        
    h = tmp # initialization matrix (2,2) for one frequency
    assert(h.shape == (2,2))
    
    I = np.eye(2) # 2x2 indentity matrix
    I = np.reshape(I, (2,2,1)) # reshape to include neuron combinations axis
    assert(I.shape == (2,2,1))  

    # make a stack of initialization matrixes for each frequency
    # the follow stack could be one too short along the freq axis
    psi = np.concatenate([h[:,:, np.newaxis] for _ in range(N2)], axis = 2) # first dimension is the frequency dimension
    assert(psi.shape == (2,2,N2))
    
    psi = np.reshape(psi, (2, 2, m, N2)) # reshape to use vectorized matrix operations   
    assert(psi.shape == (2,2,1,N2))
    
    Sarr = np.reshape(Sarr, (2, 2, m, N2)) # reshape to use vectorized matrix operations 
    assert(Sarr.shape == (2,2,1,N2))
            
    # iterative matrix factorization
    for iter in range(Niterations):
        invpsi = inv2x2(psi)
        # add the identity matrix broadcasted over the frequency axis
        g = sandwich2x2(invpsi,Sarr)+I[:, :, :, np.newaxis] 
        assert(g.shape == (2,2,1,N2))
        
        gp = _PlusOperator(g, m, N+1)        
        assert(gp.shape == (2,2,1,N2))
        
        psi_old = psi.copy()
        psi = mtimes2x2(psi, gp)
        assert(psi.shape == (2,2,1,N2))
        
        psierr = np.abs(psi-psi_old)/np.abs(psi)
        psierrf = np.mean(psierr)
        
        if psierrf < tol:
            print('reached convergence at iteration %d' % iter)
            break
            
    # Step 5: getting covariance matrix from spectral factors
    gamtmp = np.real(np.fft.ifft(psi, axis = 3)) # ifft over frequency axis, which is fourth axis

    # Step 6: getting noise covariance function and transfer function (see Example pp. 424 (of what?))
    A0 = gamtmp[:,:,:,0] # zero frequency/lag component (noise)
    assert(A0.shape == (2,2,m))
    
    A0 = np.reshape(A0, (2,2,m,1))
    A0inv = inv2x2(A0) # inverse of 2x2xmxN matrix
    
    assert(A0inv.shape == (2,2,m,1))
    A0 = A0.reshape((2,2,m))
    A0inv = A0inv.reshape((2,2,m))
    
    Z = np.zeros((2,2,m)) # container for the noise covariance
    assert(Z.shape == (2,2,1)) # test shape of covariance matrix
    for k in range(m):
        Z[:,:,k] = np.dot(A0[:,:,k], np.transpose(A0[:,:,k])) # noise covariance matrix not multiplied by the frequency
    
    H = np.zeros((2,2,m,N+1), dtype = np.complex128)
    S = np.zeros((2,2,m,N+1), dtype = np.complex128)

    for freqInd in range(N+1):
        for cmb in range(m):       
            H[:,:,cmb,freqInd] = np.dot(psi[:,:,cmb,freqInd], A0inv[:,:,cmb]) # transfer function
            S[:,:,cmb,freqInd] = np.dot(psi[:,:,cmb,freqInd], psi[:,:,cmb,freqInd].conj().T) # cross-spectral density
                           
    if m == 1: # if only one pair of neurons was analyzed
        H = H.squeeze() # remove unit dimensions
        S = S.squeeze() # remove unit dimensions
        psi = psi.squeeze() # remove unit dimensions
        Z = Z.squeeze() # remove unit dimensions
    else:
        raise ArgumentError('adjust code to work with more than one neuron pair')
 
    print('Z', Z)
    return H, Z, S, psi
            
def _PlusOperator(g, ncmb, nfreq):
    """
    This is the []+ operator as describes in wilson's paper
    here I have dropped the reshaping as in fieldtrip's code for selecting an
    axis for the fft
    
    """
    gam = np.fft.ifft(g, axis = 3)          # fourth axis is the frequency axis (chanxchanxcmbxfreq)
    assert(gam.shape == (2,2,ncmb, (2*nfreq)-2)) # test for expected size
    gamp = gam.copy()                       # copy to prevent corruption
    beta0 = 0.5*gam[:,:,:, 0]               # lowest lag component for all combinations of all channels
    assert(beta0.shape == (2,2,ncmb))       # test for expected size
    
    beta0[1, 0, :] = 0                      # set the g21 component of the 0-lag to zero
    assert(beta0[1,0,0] == 0)               # see if g21 indeed is zero
    
    gamp[:, :, :, 0] = beta0                # assign the result back to the original
    assert(np.all(gamp[:,:,:,0] == beta0))
    
    gamp[:,:,:, nfreq:] = 0               # pick only positive lags (nfreq instead of nfreq+1, due to 0-indexing)
    assert(np.all(gamp[:,:,:,nfreq:] == 0))
    
    gp = np.fft.fft(gamp, axis = 3)         # refft inverse of above ifft
    assert(gp.shape == (2,2,1,(2*nfreq)-2))
    
    return gp
