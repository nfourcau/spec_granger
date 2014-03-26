import numpy as np

def sandwich2x2(x, y):
    """
    SANDWICH2X2 compute x*y*x' provided y is Hermitian and dimensionality is 2x2xN
    
    """
    # FIXME build in check for hermitianity
    z = np.zeros(np.shape(x), dtype = np.complex)
    xconj = np.conj(x)
    xabs2 = np.abs(x)**2
    
    z[0, 0, :, :] = xabs2[0, 0, :, :] * y[0, 0, :, :] + \
                    xabs2[0, 1, :, :] * y[1, 1, :, :] + \
                    2*np.real(y[1, 0, :, :]*xconj[0, 0, :, :] * x[0, 1, :, :])
    z[1, 0, :, :] = y[0, 0, :, :] * xconj[0, 0, :, :] * x[1, 0, :, :] + \
                    y[1, 0, :, :] * xconj[0, 0, :, :] * x[1, 1, :, :] + \
                    y[0, 1, :, :] * xconj[0, 1, :, :] * x[1, 0, :, :] + \
                    y[1, 1, :, :] * xconj[0, 1, :, :] * x[1, 1, :, :]
    z[0, 1, :, :] = np.conj(z[1, 0, :, :])
    z[1, 1, :, :] = xabs2[1, 0, :, :] * y[0, 0, :, :] + \
                    xabs2[1, 1, :, :] * y[1, 1, :, :] + \
                    2 * np.real(y[0, 1, :, :] * xconj[1, 1, :, :] * x[1, 0,:, :])
                    
    return z
                    
# matlab code for sandwich
#z(1,1,:,:) = xabs2(1,1,:,:) .* y(1,1,:,:) + ...
#             xabs2(1,2,:,:) .* y(2,2,:,:) + ...
#           2.*real(y(2,1,:,:).*xconj(1,1,:,:).*x(1,2,:,:));
#z(2,1,:,:) = y(1,1,:,:).*xconj(1,1,:,:).*x(2,1,:,:) + ...
#           y(2,1,:,:).*xconj(1,1,:,:).*x(2,2,:,:) + ...
#           y(1,2,:,:).*xconj(1,2,:,:).*x(2,1,:,:) + ...
#           y(2,2,:,:).*xconj(1,2,:,:).*x(2,2,:,:);
#z(1,2,:,:) = conj(z(2,1,:,:));
#z(2,2,:,:) = xabs2(2,1,:,:) .* y(1,1,:,:) + ...
#             xabs2(2,2,:,:) .* y(2,2,:,:) + ...
#           2.*real(y(1,2,:,:).*xconj(2,2,:,:).*x(2,1,:,:));




def inv2x2(x):
    """
    % INV2X2 computes inverse of matrix x, using explicit analytic definition
    
    """
    siz = np.array(np.shape(x))
    assert(len(x.shape)==4)
        
    if np.all(siz[0:2] == 2):
        adjx = np.array([[x[1, 1, :, :], -x[0, 1, :, :]], [-x[1, 0, :, :], x[0, 0, :, :]]])
        assert(len(adjx.shape)==4)
        
        denom = det2x2(x) # shape returned has 4 dimensions
              
        denom = np.tile(denom, (2,2,1,1)) # instead of matlab based fancy indexing
        assert(len(denom.shape)==4)
        d = adjx/denom # check whether the slicing is the same for numpy TO_DO
    
    elif np.all(siz[0:2] == 3):
        adjx = np.array([[det2x2(x[[1,2], [1,2], :, :]), -det2x2(x[[0,2], [1,2], :, :]) , det2x2(x[[0, 1], [1, 2], :, :])],
        [-det2x2(x[[1,2], [0, 2], :, :]), det2x2(x[[0,2], [0,2], :, :]), -det2x2(x[[0,1], [0,2], :, :])],
        [det2x2(x[[1,2], [0,1], :, :]), -det2x2(x[[0,2], [0,1], :, :]), det2x2(x[[0,1], [0,1], :, :])]])
        
        denom = det2x2(x)
        d = adjx/denom[[0,0,0], [0,0,0], :, :]
    elif np.size(siz) == 2:
        d = inv(x)
    else:           
        raise Exception('cannot compute slicewice inverse')
           
    
    return d    
    
    
# matlab code for inv2x2
#siz = size(x);
#if all(siz(1:2)==2),
#  adjx  = [x(2,2,:,:) -x(1,2,:,:); -x(2,1,:,:) x(1,1,:,:)];
#  denom = det2x2(x);
#  d     = adjx./denom([1 1],[1 1],:,:);
#elseif all(siz(1:2)==3),
#  adjx = [ det2x2(x([2 3],[2 3],:,:)) -det2x2(x([1 3],[2 3],:,:))  det2x2(x([1 2],[2 3],:,:)); ...
#          -det2x2(x([2 3],[1 3],:,:))  det2x2(x([1 3],[1 3],:,:)) -det2x2(x([1 2],[1 3],:,:)); ...
#	   det2x2(x([2 3],[1 2],:,:)) -det2x2(x([1 3],[1 2],:,:))  det2x2(x([1 2],[1 2],:,:))];
#  denom = det2x2(x);
#  d     = adjx./denom([1 1 1],[1 1 1],:,:);
#elseif numel(siz)==2,
#  d = inv(x);
#else
#  error('cannot compute slicewise inverse');
#  % write for loop for the higher dimensions, using normal inv
#end

def det2x2(x):
    """
    computes determinant of matrix x, using explicit analytic definition if
    size(x,1) < 4, otherwise use matlab det-function
    
    """
    siz = np.array(np.shape(x))
    
    if np.all(siz[0:2]==2):
        d = x[0,0,:,:] * x[1,1,:,:] - x[0,1,:,:]*x[1,0,:,:]
    elif np.all(siz[0:2]==3):
        d = x[0,0,:,:] * x[1,1,:,:] * x[2,2,:,:] - \
        x[0,0,:,:] * x[1,2,:,:] * x[2,1,:,:] - \
        x[0,1,:,:] * x[1,0,:,:] * x[2,2,:,:] + \
        x[0,1,:,:] * x[1,2,:,:] * x[2,0,:,:] + \
        x[0,2,:,:] * x[1,0,:,:] * x[2,1,:,:] - \
        x[0,2,:,:] * x[1,1,:,:] * x[2,0,:,:]
    
    elif np.size(siz)==2:
        d = det(x)
    else:
        raise Exception('determinant cannot be computed')        
    
    #print('determinant1', d.shape)
    dre = d.reshape((1,1)+d.shape)
    assert(dre.shape==(1,1)+d.shape)
    #print('determinant2', dre.shape)    
    return dre

# matlab code of det2x2    
#siz = size(x);
#if all(siz(1:2)==2),
#  d = x(1,1,:,:).*x(2,2,:,:) - x(1,2,:,:).*x(2,1,:,:);
#elseif all(siz(1:2)==3),
#  d = x(1,1,:,:).*x(2,2,:,:).*x(3,3,:,:) - ...
#      x(1,1,:,:).*x(2,3,:,:).*x(3,2,:,:) - ...
#      x(1,2,:,:).*x(2,1,:,:).*x(3,3,:,:) + ...
#      x(1,2,:,:).*x(2,3,:,:).*x(3,1,:,:) + ...
#      x(1,3,:,:).*x(2,1,:,:).*x(3,2,:,:) - ...
#      x(1,3,:,:).*x(2,2,:,:).*x(3,1,:,:);
#elseif numel(siz)==2,
#  d = det(x);
#else
#  %error   
#  %write for loop
#  %for
#  %end
#end


def mtimes2x2(x, y):
    """
    MTIMES2X2 compute x*y where the dimensionatity is 2x2xN or 2x2xNxM
    
    """
    z = np.zeros(np.shape(x), dtype = np.complex)
    # xconj = np.conj(x)
    
    z[0,0,:,:] = x[0,0,:,:] * y[0,0,:,:] + x[0,1,:,:] * y[1,0,:,:]
    z[0,1,:,:] = x[0,0,:,:] * y[0,1,:,:] + x[0,1,:,:] * y[1,1,:,:]
    z[1,0,:,:] = x[1,0,:,:] * y[0,0,:,:] + x[1,1,:,:] * y[1,0,:,:]
    z[1,1,:,:] = x[1,0,:,:] * y[0,1,:,:] + x[1,1,:,:] * y[1,1,:,:]   

    assert(len(z.shape)==4) # check for 4D
    
    return z

# matlab code for mtimes2x2
#z     = complex(zeros(size(x)));
#xconj = conj(x);
#
#z(1,1,:,:) = x(1,1,:,:).*y(1,1,:,:) + x(1,2,:,:).*y(2,1,:,:);
#z(1,2,:,:) = x(1,1,:,:).*y(1,2,:,:) + x(1,2,:,:).*y(2,2,:,:);
#z(2,1,:,:) = x(2,1,:,:).*y(1,1,:,:) + x(2,2,:,:).*y(2,1,:,:);
#z(2,2,:,:) = x(2,1,:,:).*y(1,2,:,:) + x(2,2,:,:).*y(2,2,:,:);

