import numpy as np
import matplotlib.pyplot as plt

def reconstitute(V, L, UT, compressionrank=-1):
    """ 
        Assumes fullmatrices=True V must be mxm UT must be nxn for a mxn decomposition 
        This is Vpq,n as per paper lingo
    """
    L = L.copy()
    if compressionrank <= 0 or compressionrank >= len(L):
        pass
    else:
        L[compressionrank+1:] = 0.0 # truncate eigenvalues to simulate lossy effect of reduced rank reconstruction
    return np.dot(V[:, :len(L)] * L, UT)

def packmat(X, Xflagged, policy="square"):
    """
        Repacks input matrix X of shape M x N
        into semi-square matrix with flagged values removed
        as given by Xflagged of shape M x N

        policy can be the following:
            "square": attempt to pack into a squareish matrix
            "samecol": pack into a matrix with the same number
                       of columns as X
            "colminofcolrow": pack into a matrix with the column
                              size the minimum of the M and N

        This procedure does not change the in-memory orderinging
        of the elements, but will removed flagged values

        IMPORTANT: This procedure will remove partially filled rows,
                   some elements at the end of the ravel may be lost 
                   upon unpacking!

        returns shape of packed matrix, packed_matrix
    """
    if X.shape != Xflagged.shape:
        raise ValueError("Xflagged must be of same shape as X")
    if Xflagged.dtype != bool:
        raise ValueError("Xflagged must be boolean matrix")
    if X.ndim != 2:
        raise ValueError("X must be 2D matrix")
    if policy not in ["square", "samecol", "colminofcolrow"]:
        raise ValueError(f"Unknown repacking policy {policy}")

    raveled = X[np.logical_not(Xflagged)].ravel()
    
    if policy == "square":
        ncol = int(np.ceil(np.sqrt(raveled.size)))
        nrow = ncol # initial guess - will refine lower down
    elif policy == "samecol":
        ncol = X.shape[1]
        nrow = X.shape[0]
    elif policy == "colminofcolrow":
        ncol = min(X.shape[0], X.shape[1])
        nrow = max(X.shape[0], X.shape[1])

    assert ncol*nrow >= raveled.size
    nmissing = ncol*nrow - raveled.size
    nrow = nrow - (nmissing // ncol + (nmissing % ncol > 0)) if ncol > 0 else 0
    truncX = np.zeros((nrow, ncol), dtype=X.dtype).ravel()
    assert truncX.size <= raveled.size
    truncX[...] = raveled[:truncX.size]
    return (nrow, ncol), truncX.reshape(nrow, ncol)

def unpackmat(Xpacked, origXflags, flagged_value=0.0):
    """
        Unpacks a matrix packed by packmat routine
        Xpacked is a packed matrix P x Q
        origXflags is the flag array of shape M x N
        Note: P * Q <= M * N guarranteed by packmat routine
        Note: some values at the end of the the P * Q
        block may have been truncated by the packmat routine

        Returns flag array of shape M x N with additional flags 
        at the end of the array flagged array set where values
        were truncated by packmat routine and the unpacked data
        array of shape M x N
    """

    if Xpacked.ndim != 2:
        raise ValueError("Xpacked must be 2D matrix")
    if origXflags.ndim != 2:
        raise ValueError("origXflags must be 2D matrix")
    if Xpacked.size > origXflags.size:
        raise ValueError("Packed size exceeds requested unpacking size")
    if origXflags.dtype != bool:
        raise ValueError("origXflags must be boolean matrix")

    outflags = origXflags.copy()
    outdata = np.ones(outflags.shape, dtype=Xpacked.dtype) * flagged_value
    # packed array guarranteed to be equal or less in size
    # meaning we need to flag unflagged values from the end of the flag memory block
    # lost in the packing process
    nmissing = np.sum(np.logical_not(origXflags)) - Xpacked.size
    if nmissing < 0:
        raise ValueError("Expected Xpacked array to be smaller or equal in size to the original flags")
    selmissing =  np.argwhere(np.logical_not(outflags.ravel()))[::-1][:nmissing]
    outflags.reshape(outflags.size)[selmissing] = True
        
    selout = np.logical_not(outflags)
    assert Xpacked.size == np.sum(selout)
    outdata[selout] = Xpacked.ravel() 

    return outdata, outflags
    
if __name__=='__main__':
    # domain
    nnu = 1024
    nu = np.linspace(-np.pi, np.pi, nnu)
    nt = 128
    t = np.linspace(0, 25, nt)

    # low rank function (a single fringe)
    x = np.exp(1.0j*(nu[:, None] + t[None, :]))

    V, L, U = np.linalg.svd(x, full_matrices=False)

    xbar = reconstitute(V, L, U)
    if not np.allclose(x, xbar):
        print(np.max(np.abs(x - xbar)))

    # choose 10% random flags
    nflagged = int(0.1*nnu*nt)
    np.random.seed(0)
    Inu = np.random.randint(0, nnu, nflagged)
    It = np.random.randint(0, nt, nflagged)
    xflagged1 = x.copy()
    xflagged1[Inu, It] = 0j
    V1, L1, U1 = np.linalg.svd(xflagged1)

    # flag entire time and freq slots
    xflagged2 = x.copy()
    xflagged2[:, 25] = 0j
    xflagged2[:, 45] = 0j
    xflagged2[512, :] = 0j
    V2, L2, U2 = np.linalg.svd(xflagged2)

    # flag ravelled uniform
    xflagged3 = xflagged2.copy()
    xflagged3_flags = xflagged3 == 0j
    packedshape, pxflagged3 = packmat(xflagged3, xflagged3_flags, policy="samecol")
    V3, L3, U3 = np.linalg.svd(pxflagged3, full_matrices=True)
    recon = reconstitute(V3, L3, U3)
    unpacked3, unpacked_flags3 = unpackmat(recon, xflagged3_flags)
    seldiff = np.logical_not(unpacked_flags3)
    assert(np.allclose(unpacked3[seldiff], xflagged3[seldiff]))
    
    # flag ravelled random
    xflagged4 = xflagged1.copy()
    xflagged4_flags = xflagged4 == 0j
    packedshape, pxflagged4 = packmat(xflagged4, xflagged4_flags, policy="samecol")
    V4, L4, U4 = np.linalg.svd(pxflagged4)
    recon = reconstitute(V4, L4, U4)
    unpacked4, unpacked_flags4 = unpackmat(recon, xflagged4_flags)
    seldiff = np.logical_not(unpacked_flags4)
    assert(np.allclose(unpacked4[seldiff], xflagged4[seldiff]))

    print('Rank of unflagged matrix = ', np.linalg.matrix_rank(x))
    print('Rank of randomly flagged matrix = ', np.linalg.matrix_rank(xflagged1))
    print('Rank of uniformly flagged matrix = ', np.linalg.matrix_rank(xflagged2))
    print('Rank of uniformly flagged matrix with flags removed = ', np.linalg.matrix_rank(pxflagged3))
    print('Rank of randomly flagged matrix with flags removed = ', np.linalg.matrix_rank(pxflagged4))

    plt.figure()
    plt.plot(L, 'xr', label='orig')
    plt.plot(L1, '+k', label='random')
    plt.plot(L2, '*b', label='uniform')
    plt.plot(L3, '*m', label='uniform, flagsremoved')
    plt.plot(L4, '+k', alpha=0.5, label='random, flagsremoved')
    plt.legend()

    plt.show()