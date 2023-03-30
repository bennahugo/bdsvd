#!/usr/bin/env python3
import matplotlib
matplotlib.use("agg")

import numpy as np
import sys
import os
import functools
import shutil
import matplotlib.pyplot as plt
from casacore.tables import table as tbl
from casacore.tables import taql
from progress.bar import FillingSquaresBar as bar
import logging
import argparse

'''
Enumeration of stokes and correlations used in MS2.0 - as per Stokes.h in casacore, the rest are left unimplemented:
These are useful when working with visibility data (https://casa.nrao.edu/Memos/229.html#SECTION000613000000000000000)
'''
StokesTypes = {'I': 1, 'Q': 2, 'U': 3, 'V': 4, 'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8, 'XX': 9, 'XY': 10, 'YX': 11,
               'YY': 12}
ReverseStokesTypes = {1: 'I', 2: 'Q', 3: 'U', 4: 'V', 5: 'RR', 6: 'RL', 7: 'LR', 8: 'LL', 9: 'XX', 10: 'XY', 11: 'YX',
                      12: 'YY'}

class progress():
    def __init__(self, *args, **kwargs):
        """ Wraps a progress bar to check for TTY attachment
            otherwise does prints basic progress periodically
        """
        if sys.stdout.isatty():
            self.__progress = bar(*args, **kwargs)
        else:
            self.__progress = None
            self.__value = 0
            self.__title = args[0]
            self.__max = kwargs.get("max", 1)

    def next(self):
        if self.__progress is None:
            if self.__value % int(self.__max * 0.1) == 0:
                logger.info(f"\t {self.__title} progress: "
                            f"{self.__value *100. / self.__max:.0f}%")
            self.__value += 1
        else:
            self.__progress.next()

def baseline_index(a1, a2, no_antennae):
    """
    Computes unique index of a baseline given antenna 1 and antenna 2
    (zero indexed) as input. The arrays may or may not contain
    auto-correlations.
    There is a quadratic series expression relating a1 and a2
    to a unique baseline index(can be found by the double difference
    method)
    Let slow_varying_index be S = min(a1, a2). The goal is to find
    the number of fast varying terms. As the slow
    varying terms increase these get fewer and fewer, because
    we only consider unique baselines and not the conjugate
    baselines)
    B = (-S ^ 2 + 2 * S *  # Ant + S) / 2 + diff between the
    slowest and fastest varying antenna
    :param a1: array of ANTENNA_1 ids
    :param a2: array of ANTENNA_2 ids
    :param no_antennae: number of antennae in the array
    :return: array of baseline ids
    Note: na must be strictly greater than max of 0-indexed
          ANTENNA_1 and ANTENNA_2
    """
    if a1.shape != a2.shape:
        raise ValueError("a1 and a2 must have the same shape!")

    slow_index = np.min(np.array([a1, a2]), axis=0)

    return (slow_index * (-slow_index + (2 * no_antennae + 1))) // 2 + \
        np.abs(a1 - a2)

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("BDSVD")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, console, cfmt
log, log_console_handler, log_formatter = create_logger()

def setup_cmdargs():
    # defaults
    DDID=0
    FIELDID=0
    INPUT_DATACOL="DATA"
    OUTPUT_DATACOL="DECOMPRESSED_DATA"
    FLAGVALUE = 0.
    ANTSEL=[]
    SCANSEL=[]
    OUTPUTFOLDER="output"
    PLOTON=True
    WATERFALLON=True
    PLOTOVERRIDE=True
    IGNOREFLAGS=False
    RANKOVERRIDE=0
    SIMULATE=False
    EPSILON=None
    UPSILON=None
    BASELINE_DEPENDENT_RANK_REDUCE=False
    CHECK_TOLERANCE=1e-4
    PAINTING_POLICY="WITHMODEL"

    parser = argparse.ArgumentParser(description="BDSVD -- reference implementation for Baseline Dependent SVD-based compressor")
    parser.add_argument("VIS", type=str, help="Input database")
    parser.add_argument("--DDID", "-di", dest="DDID", type=int, default=DDID, help="Selected DDID")
    parser.add_argument("--FIELDID", "-fi", dest="FIELDID", type=int, default=FIELDID, help="Selected Field ID (not name)")
    parser.add_argument("--INPUTCOL", "-ic", dest="INPUT_DATACOL", type=str, default=INPUT_DATACOL, help="Input data column to compress")
    parser.add_argument("--OUTPUTCOL", "-oc", dest="OUTPUT_DATACOL", type=str, default=OUTPUT_DATACOL, help="Column to write rank-reduced data into")
    parser.add_argument("--FLAGVALUE", dest="FLAG_VALUE_TO_USE", type=float, default=FLAGVALUE, help="Placeholder value to use for flagged data in decompressed column")
    parser.add_argument("--SELANT", "-sa", dest="ANTSEL", type=str, nargs="*", default=ANTSEL, help="Select antenna by name. Can give multiple values or left unset to select all")
    parser.add_argument("--SELSCAN", "-ss", dest="SCANSEL", type=str, nargs="*", default=SCANSEL, help="Select scan by scan number. Can give multiple values or left unset to select all")
    parser.add_argument("--CORRSEL", "-sc", dest="CORRSEL", type=str, nargs="*", default=SCANSEL, help="Select correlations as defined in casacore Stokes.h. Can give multiple values or left unset to select all")
    parser.add_argument("--OUTPUTFOLDER", dest="OUTPUTFOLDER", type=str, default=OUTPUTFOLDER, help="Output folder to use for plots and output")
    parser.add_argument("--PLOTOFF", "-po", dest="PLOTOFF", action="store_true", default=not PLOTON, help="Do not do verbose plots")
    parser.add_argument("--WATERFALLOFF", "-wo", dest="WATERFALLOFF", action="store_true", default=not WATERFALLON, help="Skip waterfall plots, has no effect when specified PLOTOFF")
    parser.add_argument("--NOSKIPAUTO", dest="NOSKIPAUTO", action="store_true", help="Don't skip processing autocorrelations")
    parser.add_argument("--PLOTNOOVERRIDE", dest="PLOTNOOVERRIDE", action="store_true", default=not PLOTOVERRIDE, help="Do not override previous output")
    parser.add_argument("--SIMULATE", "-sim", dest="SIMULATE", action="store_true", default=SIMULATE, help="Simulate compression filtering only - don't write back to database")
    parser.add_argument("--RANKOVERRIDE", "-ro", dest="RANKOVERRIDE", type=int, default=RANKOVERRIDE, help="Override compression rank 0 < n < r on all spacings (manual simple SVD). <= 0 disables override")
    parser.add_argument("--CHECKTOL", dest="CHECKTOL", type=float, default=CHECK_TOLERANCE, help="Accepted numerical tolerance per visibility full rank reconstruction")
    parser.add_argument("--EPSILON", "-ep", dest="EPSILON", type=float, default=EPSILON, help="Maximum threshold error to tolerate")
    parser.add_argument("--UPSILON", "-up", dest="UPSILON", type=float, default=UPSILON, help="Minumum percentage signal to preserve")
    parser.add_argument("--BASELINE_DEPENDENT_RANK_REDUCE", "-bd", dest="BASELINE_DEPENDENT_RANK_REDUCE", action="store_true", default=BASELINE_DEPENDENT_RANK_REDUCE, help="Enable per baseline rank reduction (BDSVD)")
    parser.add_argument("--IGNOREFLAGS", "-if", dest="IGNOREFLAGS", action="store_true", default=IGNOREFLAGS, help="Ignores input flags - can only be used with option SIMULATE")
    parser.add_argument("--INPAINTING_POLICY", "-ipp", dest="INPAINTING_POLICY", choices=["WITHMODEL", "WITHCONSTANT", "REPACK"], default=PAINTING_POLICY,
                        help="How to deal with flagged data: inpaint it with MODEL_DATA, with constant as specified by FLAGVALUE (not recommended!) or "
                             "repack matrices with flagged values excluded (this may affect output FLAG column) and percentages")
    return parser.parse_args()

def add_column(table, col_name, like_col="DATA", like_type=None):
    """
    Lifted from ratt-ru/cubical

    Inserts a new column into the measurement set.
    Args:
        col_name (str): 
            Name of target column.
        like_col (str, optional): 
            Column will be patterned on the named column.
        like_type (str or None, optional): 
            If set, column type will be changed.
    Returns:
        bool:
            True if a new column was inserted, else False.
    """

    if col_name not in table.colnames():
        # new column needs to be inserted -- get column description from column 'like_col'
        desc = table.getcoldesc(like_col)

        desc[str('name')] = str(col_name)
        desc[str('comment')] = str(desc['comment'].replace(" ", "_"))  # got this from Cyril, not sure why
        dminfo = table.getdminfo(like_col)
        dminfo[str("NAME")] =  "{}-{}".format(dminfo["NAME"], col_name)

        # if a different type is specified, insert that
        if like_type:
            desc[str('valueType')] = like_type
        table.addcols(desc, dminfo)
        return True
    return False

def diffangles(a, b):
    """ a, b in degrees """
    return (a - b + 180) % 360 - 180

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

def compression_factor(nrow, nchan, r):
    """
        Eqn 14
        One can quite easily shown that the padded left singular, eigen values and right singular
        go from 
        M x M + M x N + N x N to
        M x r + r + r x N
        after accounting for padded zeros beteen r and min(M,N) on the eigen values
        bringing the total values to be stored to
        r(M + N + 0.5)
        where r is either the full rank or the reduced rank n as per paper
        factor of 0.5 here because the singular values are real values, not complex as the left and
        right singular values are
        return CF, origsize, newsize
    """
    origsize = nrow * nchan
    newsize = r * (nrow + nchan + 0.5)
    if newsize == 0:
        return np.inf, origsize, newsize
    else:
        return origsize / newsize, origsize, newsize

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

def find_n_simpleSVD(svds, correlation_selected, epsilon=None, upsilon=None):
    """
        Interpretation of Algorithm 1
        svds dico have V, L, UT which is the paper's "V" decomposed at full rank already
        V, L, UT must be full row decomposition
        We iterate until the Frobenius norms are close enough
        We operate on a single correlation at a time, call this multiple times for the others
        svds is a dico with
            - bli: baseline index keyed dicos with
                - bllength: baseline length
                - c: correlation label keyed dicos with keys (note label not indices)
                    - data: V L UT SVD decomposition at full rank full column
                    - flag: original flags
                    - rank: rank of L (min(M=nchan, N=nrow))
                    - shape: shape of packed data tuple(truncated_row, nchan)
                    - origshape: shape of the original data (same as uncompressed flag)
                    - reduced_rank: reduced rank recommendation for L to be added/modified 
                                    by this procedure
            - meta: special key with meta columns (future needs to reconstitute MAIN table)
    """
    if epsilon is None and upsilon is None:
        raise RuntimeError("Either epsilon or upsilon must be set")
    if epsilon is not None and upsilon is not None:
        raise RuntimeError("Only one of epsilon or upsilon must be set")
    
    n = 0
    
    while True:
        r = -1
        n += 1
        norm_diff_bl = 0
        norm_reduced_bl = 0
        norm_bl = 0
        for bli in svds.keys():
            if bli == "meta": continue
            cid = ReverseStokesTypes[correlation_selected]
            if cid not in svds[bli]:
                raise RuntimeError(f"Rank finder is being run on correlation '{cid}' is not selected by user")
            V, L, UT = svds[bli][cid]['data']
            this_bl_r = min(V.shape[0], UT.shape[0]) 
            if n >= this_bl_r: continue # bl stops contributing to the error at this point
            r = max(r, this_bl_r) # some baselines with other channel resolution or number of rows may be in the mix
                                  # will iterate to maximum rank

            assert len(L) == this_bl_r
            norm_diff_bl += np.sqrt(np.sum(L[n+1:this_bl_r]**2))
            norm_reduced_bl += np.sqrt(np.sum(L[:n]**2))
            norm_bl += np.sqrt(np.sum(L[:this_bl_r]**2))
        
        stop = (norm_diff_bl / norm_bl < epsilon if epsilon else \
                norm_reduced_bl / norm_bl > upsilon / 100.) or n > r
            
        if stop:
            for bli in svds.keys():
                if bli == "meta": continue
                svds[bli][cid]['reduced_rank'] = n
            break # while

def find_n_bdSVD(svds, correlation_selected, epsilon=None, upsilon=None):
    """
        Interpretation of Algorithm 2
        svds dico have V, L, UT which is the paper's "V" decomposed at full rank already
        V, L, UT must be full row decomposition
        We iterate until the Frobenius norms are close enough per baseline as opposed to simple SVD case
        We operate on a single correlation at a time, call this multiple times for the others
        svds is a dico as per docstring for find_n_simpleSVD
    """
    if epsilon is None and upsilon is None:
        raise RuntimeError("Either epsilon or upsilon must be set")
    if epsilon is not None and upsilon is not None:
        raise RuntimeError("Only one of epsilon or upsilon must be set")
    
    for bli in svds.keys():
        if bli == "meta": continue
        cid = ReverseStokesTypes[correlation_selected]
        if cid not in svds[bli]:
                raise RuntimeError(f"Rank finder is being run on correlation '{cid}' is not selected by user")
        n = 0
        V, L, UT = svds[bli][cid]['data']
        r = min(V.shape[0], UT.shape[0])
        assert len(L) == r
        norm_bl = np.sqrt(np.sum(L[:r]**2))
        while True:
            n += 1
            norm_diff_bl = np.sqrt(np.sum(L[n+1:r]**2))
            norm_reduced_bl = np.sqrt(np.sum(L[:n]**2))
            
            stop = (norm_diff_bl / norm_bl < epsilon if epsilon else \
                    norm_reduced_bl / norm_bl > upsilon / 100.) or n > r
            if stop:
                break
        svds[bli][cid]['reduced_rank'] = n

def data_volume_calc(num_antennas,
                     num_polarisations,
                     total_obs_hrs,
                     dump_rate,
                     num_channels,
                     auto_correlations=True,
                     nrows=None,
                     num_data_cols=1,
                     bits_per_vis = 64.0,
                     bits_exposure = 64.0,  # double precision
                     bits_per_flag = 8.0,
                     bits_per_weight = 32.0,
                     indexer_size = 32.0,
                     time_size = 64.0,  # double precision
                     flag_categories = 1,
                     uvw_size = 3 * 64.0  # double precision
                     ):
    """Calculates the size of an MSv2, considering data columns, flags
    (including flag category) and weights (including weight spectrum)
    Returns in units of bytes

    Adapted from SARAO calculator backend by the original author
    """
    # Calculate the number of rows in the MS as baselines x timestamps
    # if not specified
    if nrows is None:
        baselines = num_antennas * (num_antennas - 1) / 2.0
        if auto_correlations:
            baselines += num_antennas
        timestamps = total_obs_hrs * 3600 / dump_rate
        nrows = baselines * timestamps # * (nddid = 1) * (nfield = 1) * (nobs = 1)

    # CASA memo 229 (Measurement Set v2 specification)
    # See https://casa.nrao.edu/Memos/229.html
    # Bits used per column (including non-compulsory
    # columns currently being dumped in
    # https://github.com/ska-sa/katdal/blob/master/katdal/ms_extra.py)
    # these include
    BITS_PER_ROW = {
        # Foreign keys
        "ANTENNA1": indexer_size,
        "ANTENNA2": indexer_size,
        "ARRAY_ID": indexer_size,
        "DATA_DESC_ID": indexer_size,
        "FEED1": indexer_size,
        "FEED2": indexer_size,
        "FIELD_ID": indexer_size,
        "OBSERVATION_ID": indexer_size,
        "PROCESSOR_ID": indexer_size,
        "SCAN_NUMBER": indexer_size,
        "STATE_ID": indexer_size,
        # complex valued data
        "DATA": bits_per_vis * num_polarisations * num_channels,
        # meta data
        "EXPOSURE": bits_exposure,
        "FLAG": bits_per_flag * num_polarisations * num_channels,
        "FLAG_CATEGORY": bits_per_flag
        * flag_categories
        * num_polarisations
        * num_channels,
        "FLAG_ROW": bits_per_flag,
        "WEIGHT_SPECTRUM": bits_per_weight * num_polarisations * num_channels,
        "SIGMA_SPECTRUM": bits_per_weight * num_polarisations * num_channels,
        "INTERVAL": bits_exposure,
        "SIGMA": bits_per_weight * num_polarisations,
        "WEIGHT": bits_per_weight * num_polarisations,
        "TIME": time_size,
        "TIME_CENTROID": time_size,
        "UVW": uvw_size,
    }
    # DATA is already added
    # add flexibility to add for instance CORRECTED_DATA and MODEL_DATA
    # commonly needed in reductions
    if num_data_cols == 0:
        del BITS_PER_ROW["DATA"]
    else:
        for i in range(num_data_cols - 1):
            BITS_PER_ROW[f"EXTRACOL{i}"] = BITS_PER_ROW["DATA"]

    # Calculate the number of bits per column of data, flags, and weights
    total_bits = functools.reduce(lambda a, b: a + b, BITS_PER_ROW.values())

    total_data_volume = nrows * total_bits / 8

    return total_data_volume

def compress_datacol(VIS, DDID, FIELDID, INPUT_DATACOL, 
                     FLAGVALUE, ANTSEL, SCANSEL, CORRSEL, 
                     OUTPUTFOLDER, PLOTON, NOSKIPAUTO,
                     RANKOVERRIDE, SIMULATE, OUTPUT_DATACOL,
                     EPSILON, UPSILON, BASELINE_DEPENDENT_RANK_REDUCE,
                     WATERFALLON, CHECKTOL, IGNOREFLAGS, PAINTING_POLICY):
    """
        VIS - path to measurement set
        DDID - selected DDID to compress (determines SPW to select)
        FIELDID - selected field id to compress
        INPUT_DATACOL - data column to use for data compression
        FLAGVALUE - constant value to replace flagged values with
        ANTSEL - list of selected antennas by name
        SCANSEL - list of selected scans by index
        CORRSEL - list of selected correlations by value as defined by Stokes.h in casacore
        OUTPUTFOLDER - where to dump plots and other output
        PLOTON - whether to make verbose plots
        NOSKIPAUTO - by default skip auto correlations
        RANKOVERRIDE - implement manual baseline-homogenous SVD rank reduction 
                       (3.1 method 1 manual override) across all baselines
                       <= 0 or >= min(nchan, ncorr) to not override Algorithm 1 or 2
        SIMULATE - simulate only, don't modify database
        OUTPUT_DATACOL - output column to write rank-reduced data into
        EPSILON - per paper, maximum threshold error
        UPSILON - per paper, minimum percentage signal to preserve
        BASELINE_DEPENDENT_RANK_REDUCE - use algorithm 1 or algorithm 2 to reduce rank globally or per baseline,
                                         True to make the rank reduction baseline dependent through algorithm 2.
                                         Has no effect if RANKOVERRIDE is in effect
        WATERFALLON - make diff waterfall plots as well when PLOTON is specified
        CHECKTOL - accepted per visibility numercal tolerance for full rank reconstruction
        IGNOREFLAGS - Ignores input flags
        PAINTING_POLICY - "WITHMODEL", "WITHCONSTANT" or "REPACK". How to deal with flagged data: inpaint it with 
                         MODEL_DATA, with constant as specified by FLAGVALUE (not recommended!) or 
                         repack matrices with flagged values excluded (this may affect output FLAG column) and 
                         percentages
    """
    # domain will be nrow x nchan per correlation
    with tbl(f"{VIS}::FIELD", ack=False) as t:
        fieldnames = t.getcol("NAME")
        if FIELDID < 0 or FIELDID >= len(fieldnames):
            raise RuntimeError("Invalid selected field id {FIELDID}")
    log.info(f"Processing visibilities for '{VIS}' DDID {DDID} field '{fieldnames[FIELDID]}'")
    with tbl(f"{VIS}::ANTENNA", ack=False) as t:
        antnames = t.getcol("NAME")
        antpos = t.getcol("POSITION")
        if not ANTSEL:
            ANTSEL = antnames
        if not all(map(lambda a: a in antnames, ANTSEL)):
            raise RuntimeError(f"Invalid selection of antennae: '{','.join(ANTSEL)}'")
        selantind = list(map(lambda x: antnames.index(x), ANTSEL))

    log.info(f"Selected stations: {','.join(ANTSEL)} indices {','.join(map(str, selantind))}")
    with tbl(f"{VIS}::DATA_DESCRIPTION", ack=False) as t:
        ddidspwmap = t.getcol("SPECTRAL_WINDOW_ID")
    with tbl(f"{VIS}::SPECTRAL_WINDOW", ack=False) as t:
        spwfreqs = t.getcol("CHAN_FREQ")[ddidspwmap[DDID], :]
    with tbl(f"{VIS}::POLARIZATION", ack=False) as t:
        corrtypes = t.getcol("CORR_TYPE")[ddidspwmap[DDID], :]
        log.info(f"The following correlations are available: {','.join(map(lambda c: ReverseStokesTypes[c], corrtypes))}")
        if not CORRSEL:
            CORRSEL = corrtypes
        if not all(map(lambda c: c in corrtypes, CORRSEL)):
            raise RuntimeError(f"Invalid selection of correlations: '{','.join(map(lambda c: ReverseStokesTypes[c], CORRSEL))}'")
        log.info(f"User selected correlations are: {','.join(map(lambda c: ReverseStokesTypes[c], CORRSEL))}")
        
    with tbl(VIS, ack=False) as t:
        with taql(f"select from $t where DATA_DESC_ID=={DDID} and FIELD_ID=={FIELDID}") as tt:
            scans = sorted(map(str, np.unique(tt.getcol("SCAN_NUMBER"))))
            log.info(f"For selected DDID and field available scans are: {','.join(scans)}")
            if not SCANSEL:
                SCANSEL = scans
            if not all(map(lambda s: str(s) in scans, SCANSEL)):
                raise RuntimeError(f"Invalid selection of scans: '{','.join(map(str, SCANSEL))}'")
            log.info(f"User selected scans are: {','.join(map(str, SCANSEL))}")
    with tbl(VIS, ack=False, readonly=SIMULATE) as t:
        if not SIMULATE:
            if OUTPUT_DATACOL not in t.colnames():
                log.info(f"Column '{OUTPUT_DATACOL}' does not exist. Adding... - do NOT interrupt!")
                if not add_column(t, OUTPUT_DATACOL, like_col=INPUT_DATACOL):
                    raise RuntimeError(f"Could not add column {OUTPUT_DATACOL} to database")
                t.flush()
                log.info(f"<OK>")
            log.info(f"Initializing column '{OUTPUT_DATACOL}' to zeros - do NOT interrupt!")
            for s in SCANSEL:
                query = f"select from $t where DATA_DESC_ID=={DDID} and " \
                        f"FIELD_ID=={FIELDID} and " \
                        f"SCAN_NUMBER=={s} and " \
                        f"(ANTENNA1 in [{','.join(map(str, selantind))}] and " \
                        f" ANTENNA2 in [{','.join(map(str, selantind))}])"
                with taql(query) as tt:
                    data = tt.getcol(INPUT_DATACOL)
                    data[...] = 0 # init to 0
                    tt.putcol(OUTPUT_DATACOL, data)
                    tt.flush()
            log.info(f"<OK>")
    for s in SCANSEL:
        with tbl(VIS, ack=False, readonly=True) as t: # decomposition may take long... we want to make this interruptable
            query = f"select from $t where DATA_DESC_ID=={DDID} and " \
                    f"FIELD_ID=={FIELDID} and " \
                    f"SCAN_NUMBER=={s} and " \
                    f"(ANTENNA1 in [{','.join(map(str, selantind))}] and " \
                    f" ANTENNA2 in [{','.join(map(str, selantind))}])"
            with taql(query) as tt:
                log.info(f"Processing scan {s}. Selecting {tt.nrows()} rows from '{INPUT_DATACOL}' column")
                data = tt.getcol(INPUT_DATACOL)
                flag = tt.getcol("FLAG")
                if IGNOREFLAGS and SIMULATE: # only when simulating to prevent contaminating original flags
                    flag[...] = False
                if PAINTING_POLICY == "WITHMODEL":
                    model = tt.getcol("MODEL_DATA")
                else:
                    model = None
                a1 = tt.getcol("ANTENNA1")
                a2 = tt.getcol("ANTENNA2")
                if "WEIGHT_SPECTRUM" in tt.colnames():
                    weight = tt.getcol("WEIGHT_SPECTRUM")
                else:
                    weight = tt.getcol("WEIGHT")
                bl = baseline_index(a1, a2, max(np.max(a1), np.max(a2)))
                uniqbl = np.unique(bl)
                log.info("\tRead data")
                p = progress("\tDecomposing baselines", max=len(uniqbl))
                svds = {}
                #
                # STEP 1: decompose baselines and work out rank reductions per alg 1 or alg 2
                #
                for bli in uniqbl:
                    selbl = bl == bli
                    if np.sum(selbl) == 0:
                        p.next()
                        continue
                    if np.sum(flag[selbl]) == flag[selbl].size:
                        p.next()
                        continue
                    bla1 = a1[selbl][0]
                    bla2 = a2[selbl][0]
                    if not NOSKIPAUTO and bla1 == bla2:
                        p.next()
                        continue
                    svds[bli] = {
                        "bllength": np.sqrt(np.sum((antpos[bla1]-antpos[bla2])**2)),
                    }
                    for c in CORRSEL:
                        ci = np.where(corrtypes == c)[0][0]
                        corrlbl = ReverseStokesTypes[c]
                        seldata = data[selbl, :, ci].T.copy()
                        selflag = flag[selbl, :, ci].T.copy()
                        if PAINTING_POLICY == "WITHMODEL":
                            selmodel = model[selbl, :, ci].T.copy()
                            seldata[selflag] = selmodel[selflag]
                            packeddata = seldata
                        elif PAINTING_POLICY == "WITHCONSTANT":
                            seldata[selflag] = FLAGVALUE
                            packeddata = seldata
                        elif PAINTING_POLICY == "REPACK":
                            # we have to remove the flagged data from the mix on this
                            # before taking SVD. We will be truncating rows if needed
                            _, packeddata = packmat(seldata, selflag, policy="samecol")
                        else:
                            raise ValueError(f"Illegal option '{PAINTING_POLICY}' for PAINTING_POLICY")
                        
                        fullrank = np.linalg.matrix_rank(packeddata)
                        V, L, U = np.linalg.svd(packeddata, full_matrices=True)
                        
                        # --- sanity check --- 
                        # check that the full rank decompression yields the same values
                        # truncated to size of the packed matrix
                        reconstitution = reconstitute(V, L, U, 
                                                      compressionrank=fullrank)
                        if PAINTING_POLICY == "REPACK":
                            outdata, outflags = unpackmat(reconstitution, 
                                                          selflag,
                                                          flagged_value=FLAGVALUE)
                        else:
                            outdata = reconstitution
                            outflags = selflag
                        assert outdata.shape == seldata.shape
                        assert np.sum(selflag) <= np.sum(outflags)
                        diffsel = np.logical_not(outflags)
                        assert all(np.abs(outdata[diffsel] - seldata[diffsel]) < CHECKTOL)
                        # --- end sanity check --- 

                        svds[bli][corrlbl] = {
                            "data": (V, L, U),
                            "flag": selflag.T.copy(),
                            "rank": fullrank,
                            "shape": (U.shape[0], V.shape[0]), # D[row x chan] -> V[row x chan], U[row x chan]
                            "origshape": selflag.T.shape,
                            "reduced_rank": fullrank, # for now
                        }
                    p.next()
                log.info("<OK>")
                for c in CORRSEL:
                    cid = ReverseStokesTypes[c]
                    if EPSILON is not None or UPSILON is not None:
                        if BASELINE_DEPENDENT_RANK_REDUCE:
                            find_n_bdSVD(svds, c, EPSILON, UPSILON)
                        else:
                            find_n_simpleSVD(svds, c, EPSILON, UPSILON)
                    # override rank manually if set
                    # if neither is set then no rank reduction will be undertaken
                    for bli in svds:
                        if bli == "meta": continue
                        fullrank = svds[bli][cid]["rank"]
                        do_rankoverride = RANKOVERRIDE > 0 and RANKOVERRIDE < fullrank
                        svds[bli][cid]["reduced_rank"] = \
                            svds[bli][cid]["reduced_rank"] if not do_rankoverride else \
                                RANKOVERRIDE
                        
                        # compute norm Vpq,r - Vpq,n to indicate selected error level on this spacing
                        n = svds[bli][cid]['reduced_rank']
                        r = svds[bli][cid]['rank']
                        V, L, UT = svds[bli][cid]['data']
                        this_bl_r = min(V.shape[0], UT.shape[0]) 
                        svds[bli][cid]['rank_diff_norm'] = np.sqrt(np.sum(L[n+1:this_bl_r]**2))
                #
                # STEP 2: do rank filtering and gather some statistics
                #
                origsize = np.zeros(len(CORRSEL), dtype=np.int64)
                newsize = np.zeros(len(CORRSEL), dtype=np.int64)
                nrows = 0
                for bli in svds:
                    if bli == "meta": continue
                    selbl = bl == bli
                    bla1 = a1[selbl][0]
                    bla2 = a2[selbl][0]
                    nrows += len(bl[selbl])
                    log.info(f"\t{antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m):")
                    for cii, c in enumerate(CORRSEL):
                        ci = np.where(corrtypes == c)[0][0]
                        corrlbl = ReverseStokesTypes[c]
                        compressionrank = svds[bli][corrlbl]['reduced_rank']
                        V, L, U = svds[bli][corrlbl]['data']
                        cf, origsize_i, newsize_i = \
                            compression_factor(svds[bli][corrlbl]['shape'][0],
                                               svds[bli][corrlbl]['shape'][1],
                                               r = compressionrank)
                        origsize[cii] += origsize_i
                        newsize[cii] += newsize_i
                        compmsg = f"compressed to {compressionrank} (CF {cf:.2f})" \
                            if compressionrank > 0 and compressionrank < svds[bli][corrlbl]['rank'] else \
                            f"(compression disabled)"
                        log.info(f"\t\t{corrlbl} data rank {svds[bli][corrlbl]['rank']} "
                                 f"{compmsg}, "
                                 f"dim {'x'.join(map(str, svds[bli][corrlbl]['shape']))} "
                                 f"decompression error {svds[bli][corrlbl].get('rank_diff_norm', 0.):.2f}")
                for cii, c in enumerate(CORRSEL):
                    ci = np.where(corrtypes == c)[0][0]
                    corrlbl = ReverseStokesTypes[c]
                    corrfactor_str = f"{origsize[cii]/newsize[cii]:.3f}" if newsize[cii] > 0 else "Fully flagged"
                    log.info(f"\tData compression factor (CF) for scan {s} {corrlbl}: {corrfactor_str}")
                total_data_volume = data_volume_calc(num_antennas=len(ANTSEL),
                                                        num_polarisations=len(CORRSEL),
                                                        total_obs_hrs=0, #ignored nrow given
                                                        dump_rate=0, #ignored nrow given
                                                        num_channels=svds[bli][corrlbl]['shape'][1],
                                                        auto_correlations=NOSKIPAUTO,
                                                        nrows=nrows,
                                                        num_data_cols=0) # not storing a DATA column
                total_data_volume_with_data = data_volume_calc(num_antennas=len(ANTSEL),
                                                               num_polarisations=len(CORRSEL),
                                                               total_obs_hrs=0, #ignored nrow given
                                                               dump_rate=0, #ignored nrow given
                                                               num_channels=svds[bli][corrlbl]['shape'][1],
                                                               auto_correlations=NOSKIPAUTO,
                                                               nrows=nrows,
                                                               num_data_cols=1) # not storing a DATA column
                data_size = total_data_volume_with_data - total_data_volume
                data_corr_size = data_size / len(CORRSEL)
                compressed_data_size = np.sum(np.ceil(data_corr_size * (newsize / origsize)))
                log.info(f"\tTotal new {nrows} row MAIN table volume: "
                         f"{total_data_volume+compressed_data_size:.0f} bytes")
                #
                # STEP 3: decompress and make some waterfall and rank plots if wanted
                #
                if PLOTON:
                    p = progress("\tCreating plots", max=len(uniqbl)*2)
                    bllengths = []
                    blnorms = {}
                    for bli in svds:
                        if bli == "meta": 
                            p.next()
                            continue
                        bllengths.append(svds[bli]['bllength'])
                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            corrlbl = ReverseStokesTypes[c]
                            blnorms.setdefault(corrlbl, []).append(svds[bli][corrlbl]['rank_diff_norm'])
                        p.next()
                    plotmarkers = ['xr', '.b', 'dm', '*g'] # up to four supported
                    plt.figure()
                    for ci, c in enumerate(blnorms.keys()):
                        plt.plot(bllengths, blnorms[c], plotmarkers[ci % len(plotmarkers)], label=c)
                    plt.xlabel("Baseline length [m]")
                    plt.ylabel("$||\mathbf{V}_{pq,r} - \mathbf{V}_{pq,n}||_F$")
                    plt.xscale("log")
                    plt.title("Decompresion error vs. baseline length")
                    plt.legend()
                    imname = f"norm.vs.bllength.{VIS}.scan.{s}.png"
                    plt.savefig(os.path.join(OUTPUTFOLDER, imname))
                    plt.close()
                    for bli in svds:
                        if bli == "meta": 
                            p.next()
                            continue
                        selbl = bl == bli
                        bla1 = a1[selbl][0]
                        bla2 = a2[selbl][0]
                        plt.figure()
                        cutofflines = ["r", "b", "m", "g"]
                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            corrlbl = ReverseStokesTypes[c]
                            V, L, U = svds[bli][corrlbl]['data']
                            plt.plot(L, plotmarkers[ci % len(plotmarkers)], label=corrlbl)
                            if svds[bli][corrlbl]['reduced_rank'] > 0 and \
                                svds[bli][corrlbl]['reduced_rank'] < svds[bli][corrlbl]['rank']:
                                plt.axvline(svds[bli][corrlbl]['reduced_rank'],
                                            linewidth=3,
                                            linestyle="--",
                                            color=cutofflines[ci % len(cutofflines)],
                                            label=f"cutoff {corrlbl}")
                        plt.yscale("log")
                        plt.xlabel("Singular values")
                        plt.ylabel("Weight")
                        plt.title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                        plt.legend()
                        imname = f"{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                        plt.savefig(os.path.join(OUTPUTFOLDER, imname))
                        plt.close()
                        if WATERFALLON:
                            for c in CORRSEL:
                                ci = np.where(corrtypes == c)[0][0]
                                corrlbl = ReverseStokesTypes[c]
                                origdata = data[selbl, :, ci].T.copy()
                                origflag = flag[selbl, :, ci].T.copy()
                                V, L, U = svds[bli][corrlbl]['data']
                                reconstitution = reconstitute(V, L, U, 
                                                            compressionrank=svds[bli][corrlbl]['reduced_rank'])
                                
                                if PAINTING_POLICY == "REPACK":
                                    outdata, outflags = unpackmat(reconstitution, 
                                                                  svds[bli][corrlbl]['flag'].T.copy(),
                                                                  flagged_value=FLAGVALUE)
                                else:
                                    outdata = reconstitution
                                    outflags = svds[bli][corrlbl]['flag'].T.copy()
                                
                                origdata[origflag] = np.nan
                                outdata[outflags] = np.nan
                                
                                fig, axs = plt.subplots(3, figsize=(6,18))
                                imdiff = axs[0].imshow(np.abs(origdata - outdata), aspect='auto')
                                cbardiff = fig.colorbar(imdiff, ax=axs[0], orientation='vertical')
                                cbardiff.set_label("Abs difference")
                                axs[0].set_ylabel("Channel")
                                axs[0].set_title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                                imorig = axs[1].imshow(np.abs(origdata), aspect='auto')
                                cbardiff = fig.colorbar(imorig, ax=axs[1], orientation='vertical')
                                cbardiff.set_label("Original data")
                                axs[1].set_ylabel("Channel")
                                imrecon = axs[2].imshow(np.abs(outdata), aspect='auto')
                                cbardiff = fig.colorbar(imrecon, ax=axs[2], orientation='vertical')
                                cbardiff.set_label("Reconstructed data")
                                axs[2].set_ylabel("Channel")
                                axs[2].set_xlabel("Sample")
                                imname = f"reconstructed.{corrlbl}.amp.{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                                
                                fig.savefig(os.path.join(OUTPUTFOLDER, imname))
                                plt.close(fig)
                                
                                fig, axs = plt.subplots(3, figsize=(6,18))
                                imdiff = axs[0].imshow(diffangles(np.rad2deg(np.angle(origdata)), 
                                                                np.rad2deg(np.angle(outdata))), 
                                                    aspect='auto')
                                cbardiff = fig.colorbar(imdiff, ax=axs[0], orientation='vertical')
                                cbardiff.set_label("Phase difference [deg]")
                                axs[0].set_ylabel("Channel")
                                axs[0].set_title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                                imorig = axs[1].imshow(np.rad2deg(np.angle(origdata)), aspect='auto')
                                cbardiff = fig.colorbar(imorig, ax=axs[1], orientation='vertical')
                                cbardiff.set_label("Original data [deg]")
                                axs[1].set_ylabel("Channel")
                                imrecon = axs[2].imshow(np.rad2deg(np.angle(outdata)), aspect='auto')
                                cbardiff = fig.colorbar(imrecon, ax=axs[2], orientation='vertical')
                                cbardiff.set_label("Reconstructed data [deg]")
                                axs[2].set_ylabel("Channel")
                                axs[2].set_xlabel("Sample")
                                imname = f"reconstructed.{corrlbl}.phase.{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                                
                                fig.savefig(os.path.join(OUTPUTFOLDER, imname))
                                plt.close(fig)
                        p.next()
                    log.info("<OK>")

        #
        # STEP 4: finally write back decompressed values if needed
        #
        if not SIMULATE:
            log.info(f"\tWriting back rank decompressed data to {OUTPUT_DATACOL}... - do NOT interrupt")
            with tbl(VIS, ack=False, readonly=SIMULATE) as t:
                query = f"select from $t where DATA_DESC_ID=={DDID} and " \
                        f"FIELD_ID=={FIELDID} and " \
                        f"SCAN_NUMBER=={s} and " \
                        f"(ANTENNA1 in [{','.join(map(str, selantind))}] and " \
                        f" ANTENNA2 in [{','.join(map(str, selantind))}])"
                p = progress("\tDecompressing and writing", max=len(svds.keys()))
                with taql(query) as tt:
                    # other ways to do this but we're just going to get the shape from data
                    data = tt.getcol(INPUT_DATACOL)
                    flag = tt.getcol("FLAG")
                    for bli in svds:
                        if bli == "meta": 
                            p.next()
                            continue
                        selbl = bl == bli
                        bla1 = a1[selbl][0]
                        bla2 = a2[selbl][0]
                        # IMPORTANT:: if correlations are unselected we treat them as a unity operation
                        # uncompressed data is copied to those spots
                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            corrlbl = ReverseStokesTypes[c]
                            V, L, U = svds[bli][corrlbl]['data']

                            # --- sanity check --- 
                            # check that the full rank decompression yields the same values
                            # truncated to size of the packed matrix
                            reconstitution = reconstitute(V, L, U, 
                                                          compressionrank=svds[bli][corrlbl]['rank'])
                            if PAINTING_POLICY == "REPACK":
                                outdata, outflags = unpackmat(reconstitution, 
                                                              svds[bli][corrlbl]['flag'].T.copy(),
                                                              flagged_value=FLAGVALUE)
                            else:
                                outdata = reconstitution
                                outflags = svds[bli][corrlbl]['flag'].T.copy()
                            assert outdata.shape == data[selbl,:,ci].T.shape
                            assert np.sum(flag[selbl,:,ci]) <= np.sum(outflags)
                            diffsel = np.logical_not(outflags)
                            assert all(np.abs(outdata[diffsel] - data[selbl,:,ci].T[diffsel]) < CHECKTOL)
                            # --- end sanity check ---

                            reconstitution = reconstitute(V, L, U, compressionrank=svds[bli][corrlbl]['reduced_rank'])
                            outdata, outflags = unpackmat(reconstitution, 
                                                          svds[bli][corrlbl]['flag'].T.copy(),
                                                          flagged_value=FLAGVALUE)
                            outdata[outflags] = FLAGVALUE
                            data[selbl,:,ci] = outdata.T.copy() # we transposed earlier
                            flag[selbl,:,ci] = outflags.T.copy()
                        p.next()
                    tt.putcol(OUTPUT_DATACOL, data)
                    if PAINTING_POLICY == "REPACK":
                        tt.putcol("FLAG", flag)
                    tt.flush()
                log.info("\t<OK>")
                if PAINTING_POLICY == "REPACK":
                    log.info("Repacked matrices. May increase flagging")
                else:
                    log.info("Inpainting instead of repacking. Will not touch flags")
        else:
            log.info(f"\tSimulation of compression of scan {s} done")

if __name__=='__main__':
    args = setup_cmdargs()
    log.info(':::BDSVD running with the following parameters:::\n'+
         '\n'.join(f'{k.ljust(30, " ")} = {v}' for k, v in vars(args).items())+
         "\n === BDSVD ===")
    VIS=args.VIS
    DDID=args.DDID
    FIELDID=args.FIELDID
    INPUT_DATACOL=args.INPUT_DATACOL
    FLAGVALUE=args.FLAG_VALUE_TO_USE
    ANTSEL=args.ANTSEL
    SCANSEL=args.SCANSEL
    CORRSEL=list(map(lambda c: StokesTypes.get(c, "INVALID"), args.CORRSEL))
    OUTPUTFOLDER=args.OUTPUTFOLDER
    PLOTON=not args.PLOTOFF
    WATERFALLON=not args.WATERFALLOFF
    PLOTOVERRIDE=not args.PLOTNOOVERRIDE
    NOSKIPAUTO = args.NOSKIPAUTO
    RANKOVERRIDE = args.RANKOVERRIDE
    SIMULATE = args.SIMULATE
    OUTPUT_DATACOL = args.OUTPUT_DATACOL
    EPSILON = args.EPSILON
    UPSILON = args.UPSILON
    BASELINE_DEPENDENT_RANK_REDUCE = args.BASELINE_DEPENDENT_RANK_REDUCE
    CHECKTOL = args.CHECKTOL
    IGNOREFLAGS = args.IGNOREFLAGS
    INPAINTINGPOLICY = args.INPAINTING_POLICY
    if EPSILON is not None and UPSILON is not None:
        raise RuntimeError("Only one of epsilon or upsilon must be set")

    if PLOTON and os.path.exists(OUTPUTFOLDER) and not os.path.isdir(OUTPUTFOLDER):
        raise RuntimeError(f"Output path '{OUTPUTFOLDER}' exists, but is not a folder. Cannot take")
    if PLOTON and PLOTOVERRIDE and os.path.exists(OUTPUTFOLDER):
        shutil.rmtree(OUTPUTFOLDER)
    if PLOTON and not os.path.exists(OUTPUTFOLDER):
        os.mkdir(OUTPUTFOLDER)
    if IGNOREFLAGS and not SIMULATE:
        raise RuntimeError("Can only specify IGNOREFLAGS when specified SIMULATE")
    
    compress_datacol(VIS,
                     DDID,
                     FIELDID,
                     INPUT_DATACOL,
                     FLAGVALUE,
                     ANTSEL,
                     SCANSEL,
                     CORRSEL,
                     OUTPUTFOLDER,
                     PLOTON,
                     NOSKIPAUTO,
                     RANKOVERRIDE,
                     SIMULATE,
                     OUTPUT_DATACOL,
                     EPSILON,
                     UPSILON,
                     BASELINE_DEPENDENT_RANK_REDUCE,
                     WATERFALLON,
                     CHECKTOL,
                     IGNOREFLAGS,
                     INPAINTINGPOLICY)
    log.info("Program ending")