import numpy as np
from numpy.linalg import LinAlgError
from itertools import combinations

def NPSpectralGranger(S, freqs, mode='bivariate', pair_list=[], **kwargs):
    """
    Compute the Spectral Granger causality either in bivariate or multivariate or conditional cases using a non parametric method based on Wilson decomposition of the spectral matrix
    (see Dhamala et al. 2008 for details)
    
    Input:
    S : 2-sided spectral matrices (np.ndarray of shape freq x channel x channel)

    freqs : vector of frequency where the spectral matrices was computed (positive frequencies only)

    mode : 
    * 'bivariate' : extraction of 2X2 spectral matrices and computation of spectral Granger causality for each pair
    * 'multivariate' : spectral factorization is computed on the whole spectral matrix, 
                            and Granger causality is then computed from 2x2 extractions of the factorized matrix.
    * 'conditional' : all channels which are not part of the pair under study are considered as a "conditional signal"

    pair_list : list of tuples with channel pair indices for which Granger causality (both direction) will be computed. If the list is empt
    y, all possible pairs are computed
    
    kwargs : are used to pass Wilson algorithm parameters, in particular "Niterations"

    Output:
    Granger : an array of 'object'. Granger[n1,n2] is the Granger spectrum of n2 influencing n1. It is None is this particular pair has not been computed
    """
    nchan=S.shape[1]
    Granger=np.empty((nchan,nchan),dtype='object') # Better than a sparse matrix ???
    if len(pair_list)==0:
        pair_list=combinations(range(nchan),2)
    if mode=='bivariate':
        for pair in pair_list:
            #~ print "Pair : ",pair
            for directedPair, freqs, granger, Sbis, H, Z in _NPBivariateGranger(S[np.ix_(range(S.shape[0]),pair,pair)],freqs,**kwargs):
                Granger[pair[directedPair[0]],pair[directedPair[1]]]=granger
    elif mode == 'multivariate':
        H, Z, Sbis, psi = _FactorWilsonMultivariate(S, freqs,**kwargs)
        for pair in pair_list:
            indsf=np.ix_(range(Sbis.shape[0]),pair,pair)
            inds=np.ix_(pair,pair)
            for directedPair, freqs, granger in _ComputeGranger(H[indsf],Z[inds],Sbis[indsf],psi[indsf],freqs):
                Granger[pair[directedPair[0]],pair[directedPair[1]]]=granger
    elif mode=='conditional':
        for pair in pair_list:
            cond_channels=np.arange(nchan)
            cond_channels=cond_channels[(cond_channels!=pair[0])&(cond_channels!=pair[1])]
            #~ print "Condition : ",pair," on ",cond_channels
            Granger[pair[0],pair[1]]=_NPConditionalGranger(S, freqs, source_channel=pair[1], target_channel=pair[0], conditional_channels=cond_channels,**kwargs)
            Granger[pair[1],pair[0]]=_NPConditionalGranger(S, freqs, source_channel=pair[0], target_channel=pair[1], conditional_channels=cond_channels,**kwargs)
    return Granger


def _NPBivariateGranger(S, freqs,**kwargs):
    """
    compute the granger causality for two neurons
    
    param:
        S: 3D auto/cross spectral density matrix with chanxchanxfreq we focus on [[S11, S12],[S21,S22]] per frequency
        freqs: frequenties at which the auto/cross spectral densities were computed
        
    """
    H, Z, S, psi = _FactorWilsonMultivariate(S, freqs,**kwargs)
    
    for directedPair, freqs, granger in _ComputeGranger(H, Z, S, psi, freqs):
        yield directedPair, freqs, granger, S, H, Z

def _ComputeGranger(H, Z, S, psi, freqs):
    """
    Receive the noise covariance matrix: Z, the cross-spectral density: S (1-sided)
    the transfer function H, and the left-spectral factor psi and computes the directed granger causality
    for two orientations of the given pair
    param: 
        Z: noise covariance matrix freqx2x2: [[Z11,Z12],[Z21,Z22]]
        H: transfer function freqx2x2: [[H11, H12],[H21,H22]]
        S: spectral densities [[S11, S12],[S21, S22]]
        psi: spectral factors
    yields:
        directedPair: pair (chanA, chanB) with the direction defined from chanA -> chanB 
        freqs: frequencies at which the granger is evaluated
        granger: granger causality
    
    """
    out=[]
    for pair in ((0,1), (1,0)): # flip the direction of the granger to get the other direction
        granger = np.log(np.abs(S[:,pair[0], pair[0]]) /
        (np.abs(S[:,pair[0], pair[0]])-((Z[pair[1],pair[1]]-((Z[pair[0], pair[1]]**2) / Z[pair[0],pair[0]]))*(np.abs(H[:,pair[0],pair[1]])**2))))

        out.append([pair, freqs, granger.astype('float64')])
        
    return out


    
def _NPConditionalGranger(S, freqs, source_channel, target_channel, conditional_channels=[],**kwargs):
    """
    Compute the non parameteric spectral Granger causality.
    
    param:
        S: 3D spectral density matrix (two sided) freqxchanxchan
        freqs : frequency where S has been computed (only positive values)
        source_channel, target_channel : channel indices where Conditional Granger will be tested : source -> target
        conditional_channels : channel over which causality is conditioned
        
    output
        granger : granger causality over freqs
    """
    # To check : one sided vs two sided spectral density and transfer matrices, vs Granger spectra computation...
    
    if len(conditional_channels)==0:
        raise ExceptionError("Absence of conditional channels in ConditionalGranger not implemented")
    
    # Compute factorization of the spectral matrix for the trivariate model
    ind_trivariate=np.r_[target_channel,source_channel,conditional_channels]
    S_trivariate=S[np.ix_(np.arange(S.shape[0]),ind_trivariate,ind_trivariate)]
    Htri, Ztri, Stri, psitri=_FactorWilsonMultivariate(S_trivariate,freqs,**kwargs)
    
    # Compute factorization of the spectral matrix for the bivariate model
    ind_bivariate=np.r_[target_channel,conditional_channels]
    S_bivariate=S[np.ix_(np.arange(S.shape[0]),ind_bivariate,ind_bivariate)]
    Hbi, Zbi, Sbi, psibi=_FactorWilsonMultivariate(S_bivariate,freqs,**kwargs)

    # Compute normalization matrices
    fac1=-Ztri[1,0]/Ztri[0,0]
    fac2=-Ztri[2:,0]/Ztri[0,0]
    
    P1l=np.eye(ind_trivariate.size)
    P1l[1,0]=fac1
    P1l[2:,0]=fac2
    
    P1r=np.eye(ind_trivariate.size)
    P1r[2:,1]=-(Ztri[2:,1]+fac2*Ztri[0,1])/(Ztri[1,1]+fac1*Ztri[0,1]) # Checked. Index error in Rangarajan-Ding-2013, OK in Ding-Bressler-2006
    
    P1=np.dot(P1l,P1r)
    
    P2=np.eye(ind_bivariate.size)
    P2[1:,0]=-Zbi[1:,0]/Zbi[0,0]

    # Normalize transfer and covariance arrays
    invP1,invP2=np.linalg.inv(P1),np.linalg.inv(P2)
    normHtri=np.einsum('ijk,kl->ijl',Htri,invP1) # Htri.invP1
    normHbi=np.einsum('ijk,kl->ijl',Hbi,invP2) # Hbi.invP2
    normZtri=np.dot(np.dot(P1,Ztri),P1.T)
    normZbi=np.dot(np.dot(P2,Zbi),P2.T)
    
    # Compute spectral conditional Granger
    # TODO : make the exact computation of needed term to not have to compute the full inverse and dot product
    extHbi=np.zeros(normHtri.shape,dtype=np.complex128)
    extHbi[:,0,0]=normHbi[:,0,0]
    extHbi[:,1,1]=1
    extHbi[:,2:,0]=normHbi[:,1:,0]
    extHbi[:,0,2:]=normHbi[:,0,1:]
    extHbi[:,2:,2:]=normHbi[:,1:,1:]
    Q=np.einsum('ijk,ikl->ijl',np.linalg.inv(extHbi),normHtri) # extHbi.inv().normHtri
    
    granger=np.log(np.abs(Zbi[0,0])/np.abs(Q[:,0,0]*Q[:,0,0].conj()*Ztri[0,0]))
    
    return granger

def _FactorWilsonMultivariate(S, freqs, Niterations = 100, tol = 1e-19, init = 'chol'):
    """
    based on the sfactorwilson.m from the feartrip toolbox
    matlab param:
        tol, init: have to with the taming and initialization of the algorithm
    
    param:
        S: 3D matrix freqxchanxchan with two-sided spectral density
        fs: (sampling frequency in Hz) isn't used here I think
        freqs: vector of frequencies at which the spectral densities are given  (only positive freqs) ###### BE CAREFUL BIG CHANGES IN SPECTRAL MATRIX SHAPE COMPARED TO PREVIOUS VERSION #####
        Niterations: number of repeats 
    output:
        H: (transfer function)
        Z: (noise covariance)
        S: (cross-spectral density 1-sided)
        psi: (left spectral factor)

        # Following is to remember details of fieldtrip implementation
        usage in feartrip, they get the auto/cross spectra with ft_freqanalysis
        Here they are putting multiple timeslices through the algorithm kkkkkkk
        [Htmp, Ztmp, Stmp] = sfactorization_wilson2x2(freq.crsspctrm(:,:,:,kk), ...
                      freq.freq, numiteration, tol, cmbindx, fb, init);
     
    """
    # TODO check the various number here
    m = S.shape[1]                           # number of channels
    N = np.size(freqs)-1            # why minus 1 (originally np.size(freqs)-1
    #~ N2 = (2*N)                      # (originally 2*N with N=np.size(freqs)-1) for the Sarr matrix N2 is most likely 1 to low
    #~ assert(N2 == (2*np.size(freqs))-2)  # check of frequency element size
    N2 = S.shape[0] # Number of freqs where the two-sided spectral matrix has been computed
        
    # Step 1: forming 2-sided spectral densities for ifft routine in matlab (necessary in initial fieldtrip implementation)
    Sarr = S.copy() # data already is double sided and the right orientation (I assume)

    # Step 2: computing covariance matrices
    gam = np.real(np.fft.ifft(Sarr, axis = 0)) # ifft directly over the last axis
    gam0 = gam[0,:,:].copy() # auto/cross spectral densities of the lowest frequency
    
    h = np.zeros(gam0.shape, dtype = np.complex128) # mxm matrix used to store initial condition for optimization 
    if init == 'chol': # based on the cholesky decompositon
        try: # see if decompisition worked
            tmp = np.transpose(np.linalg.cholesky(gam0)) # numpy cholesky returns lower triangle, so transpose

        except LinAlgError: # if decomposition failed do random initialization
            print "Random initialization of Wilson algo"
            tmp = np.random.randn(m,1000) # arbitrary initial condition
            tmp = np.dot(tmp,tmp.transpose())/1000.
            tmp = np.transpose(np.linalg.cholesky(tmp))
            tmp = tmp[:,:,np.newaxis]
    elif init == 'rand': # random initial conditions
        print "Random initialization of Wilson algo"
        tmp = randn(m,1000) # arbitrary initial condition
        tmp = np.dot(tmp,tmp.transpose())/1000.
        tmp = np.transpose(np.linalg.cholesky(tmp))
        tmp = tmp[:,:,np.newaxis]
    else:
        assert(init not in ['chol', 'rand'])
        raise ArgumentError("initialization option either 'chol' or 'rand'")
    
    h[:,:] = tmp # initialization matrix (m,m) for one frequency
    
    # make a stack of initialization matrixes for each frequency
    # the follow stack could be one too short along the freq axis
    psi = np.concatenate([h[np.newaxis,:,: ] for _ in range(N2)], axis = 0) # dimension 0 is the frequency dimension
                
    # iterative matrix factorization
    for iter in range(Niterations):
                        
        # Simultaneous matrix inversion
        invpsi=np.linalg.inv(psi)
        
        # TODO Check if both einsum could be done in a single one !!!!
        g=np.einsum('ijk,ilk->ijl',np.einsum('ijk,ikl->ijl',invpsi,Sarr),invpsi.conj())+np.tile(np.eye(m),(N2,1,1)) #  invpsi.Sarr.(invpsi*.transpose()) + eye
        assert(g.shape == (N2,m,m))
        
        gp = _PlusOperator(g, N+1)        
        
        psi_old = psi.copy()
        psi=np.einsum('ijk,ikl->ijl',psi,gp) # psi;gp
        psierr=(np.abs(psi-psi_old)).sum(axis=2).sum(axis=1)
        assert(psi.shape == (N2,m,m))
        
        psierrf = np.mean(psierr)
        
        if psierrf < tol:
            print('reached convergence at iteration %d' % iter)
            break
            
    # Step 5: getting covariance matrix from spectral factors
    gamtmp = np.real(np.fft.ifft(psi, axis = 0)) # ifft over frequency axis, which is fourth axis

    # Step 6: getting noise covariance function and transfer function (see Example pp. 424 (of what?))
    A0 = gamtmp[0,:,:] # zero frequency/lag component (noise)
    A0inv = np.linalg.inv(A0) # inverse of mxm matrix
    Z = np.dot(A0, A0.transpose()) # noise covariance matrix not multiplied by the frequency

    H=np.einsum('ijk,kl->ijl',psi[:N+1,:,:],A0inv) # psi.AOinv
    S=np.einsum('ijk,ilk->ijl',psi[:N+1,:,:],psi[:N+1,:,:].conj()) # psi.(psi*.transpose())
                            
    #~ print('Z', Z)
    return H, Z, S, psi
            

def _PlusOperator(g, nfreq):
    """
    This is the []+ operator as describes in wilson's paper
    here I have dropped the reshaping as in fieldtrip's code for selecting an
    axis for the fft
    
    """
    m=g.shape[1]
    gam = np.fft.ifft(g, axis = 0)          # first axis is the frequency axis (freqxchanxchan)
    gamp = gam.copy()                       # copy to prevent corruption
    beta0 = 0.5*gam[0,:,:]               # lowest lag component for all combinations of all channels   
    beta0 = np.triu(beta0)                      # set the lower triangle components of the 0-lag to zero (this is the Stau of Wilson algorithm ???)
    gamp[0,:, :] = beta0                # assign the result back to the original    
    gamp[nfreq:,:,:] = 0               # pick only positive lags (nfreq instead of nfreq+1, due to 0-indexing)    
    gp = np.fft.fft(gamp, axis = 0)    # refft inverse of above ifft
    return gp
