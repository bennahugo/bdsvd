#!/usr/bin/env python3
import matplotlib
matplotlib.use("agg")

import numpy as np
import sys
import os
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
    PLOTOVERRIDE=True
    RANKOVERRIDE=0
    SIMULATE=False

    parser = argparse.ArgumentParser(description="BDSVD -- reference implementation for Baseline Dependent SVD-based compressor")
    parser.add_argument("VIS", type=str, help="Input database")
    parser.add_argument("--DDID", "-di", dest="DDID", type=int, default=DDID, help="Selected DDID")
    parser.add_argument("--FIELDID", "-fi", dest="FIELDID", type=int, default=FIELDID, help="Selected Field ID (not name)")
    parser.add_argument("--INPUTCOL", "-ic", dest="INPUT_DATACOL", type=str, default=INPUT_DATACOL, help="Input data column to compress")
    parser.add_argument("--OUTPUTCOL", "-oc", dest="OUTPUT_DATACOL", type=str, default=OUTPUT_DATACOL, help="Column to write rank-reduced data into. Will not affect set flags")
    parser.add_argument("--FLAGVALUE", dest="FLAG_VALUE_TO_USE", type=float, default=FLAGVALUE, help="Placeholder value to use for flagged data in input column")
    parser.add_argument("--SELANT", "-sa", dest="ANTSEL", type=str, nargs="*", default=ANTSEL, help="Select antenna by name. Can give multiple values or left unset to select all")
    parser.add_argument("--SELSCAN", "-ss", dest="SCANSEL", type=str, nargs="*", default=SCANSEL, help="Select scan by scan number. Can give multiple values or left unset to select all")
    parser.add_argument("--CORRSEL", "-sc", dest="CORRSEL", type=str, nargs="*", default=SCANSEL, help="Select correlations as defined in casacore Stokes.h. Can give multiple values or left unset to select all")
    parser.add_argument("--OUTPUTFOLDER", dest="OUTPUTFOLDER", type=str, default=OUTPUTFOLDER, help="Output folder to use for plots and output")
    parser.add_argument("--PLOTOFF", "-po", dest="PLOTOFF", action="store_true", default=not PLOTON, help="Do not do verbose plots")
    parser.add_argument("--NOSKIPAUTO", dest="NOSKIPAUTO", action="store_true", help="Don't skip processing autocorrelations")
    parser.add_argument("--PLOTNOOVERRIDE", dest="PLOTNOOVERRIDE", action="store_true", default=not PLOTOVERRIDE, help="Do not override previous output")
    parser.add_argument("--SIMULATE", "-sim", dest="SIMULATE", action="store_true", default=SIMULATE, help="Simulate compression filtering only - don't write back to database")
    parser.add_argument("--RANKOVERRIDE", "-ro", dest="RANKOVERRIDE", type=int, default=RANKOVERRIDE, help="Override compression rank 0 < n <= r on all spacings (manual simple SVD). <= 0 disables override")

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
    """ Assumes fullmatrices=True V must be mxm UT must be nxn for a mxn decomposition """
    L = L.copy()
    if compressionrank <= 0 or compressionrank >= len(L):
        pass
    else:
        L[compressionrank+1:] = 0.0 # truncate eigenvalues to simulate lossy effect of reduced rank reconstruction
    return np.dot(V[:, :len(L)] * L, UT)
    
def compress_datacol(VIS, DDID, FIELDID, INPUT_DATACOL, 
                     FLAGVALUE, ANTSEL, SCANSEL, CORRSEL, 
                     OUTPUTFOLDER, PLOTON, NOSKIPAUTO,
                     RANKOVERRIDE, SIMULATE, OUTPUT_DATACOL):
    """
        VIS - path to measurement set
        DDID - selected DDID to compress (determines SPW to select)
        FIELDID - selected field id to compress
        INPUT_DATACOL - data column to use for data compression
        FLAGVALUE - constant value to replace flagged values with for SVD purposes this must be finite
        ANTSEL - list of selected antennas by name
        SCANSEL - list of selected scans by index
        CORRSEL - list of selected correlations by value as defined by Stokes.h in casacore
        OUTPUTFOLDER - where to dump plots and other output
        PLOTON - whether to make verbose plots
        NOSKIPAUTO - by default skip auto correlations
        RANKOVERRIDE - implement manual baseline-homogenous SVD rank reduction 
                       (3.1 method 1 manual override) across all baselines
        SIMULATE - simulate only, don't modify database
        OUTPUT_DATACOL - output column to write rank-reduced data into, will not affect changes to FLAG
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
                data[flag] = FLAGVALUE # set to a constant value - cannot be nan
                uvw = tt.getcol("UVW")
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
                # STEP 1: decompose baselines
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
                        seldata = data[selbl, :, ci].T.copy()
                        V, L, U = np.linalg.svd(seldata)
                        fullrank = np.linalg.matrix_rank(seldata)
                        svds[bli][ReverseStokesTypes[c]] = {
                            "data": (V, L, U),
                            "rank": fullrank,
                            "shape": (V.shape[0], U.shape[0]), # D[chan x row] -> V[chan x row], U[chan x row]
                            # for now only have the manual baseline-homogenous rank reduction
                            # BH TODO: rank finding algorithms need to be called here
                            "reduced_rank": RANKOVERRIDE if RANKOVERRIDE > 0 else fullrank
                        }
                    p.next()
                log.info("<OK>")
                #
                # STEP 2: do rank filtering and gather some statistics
                #
                for bli in svds:
                    if bli == "meta": continue
                    selbl = bl == bli
                    bla1 = a1[selbl][0]
                    bla2 = a2[selbl][0]
                    log.info(f"\t{antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m):")
                    for c in CORRSEL:
                        ci = np.where(corrtypes == c)[0][0]
                        corrlbl = ReverseStokesTypes[c]
                        compressionrank = svds[bli][corrlbl]['reduced_rank']
                        V, L, U = svds[bli][corrlbl]['data']
                        compmsg = f"compressed to {compressionrank}" \
                            if compressionrank > 0 and compressionrank < svds[bli][corrlbl]['rank'] else \
                            f"(compression disabled)"
                        log.info(f"\t\t{corrlbl} data rank {svds[bli][corrlbl]['rank']} "
                                 f"{compmsg}, "
                                 f"dim {'x'.join(map(str, svds[bli][corrlbl]['shape']))}")
                #
                # STEP 3: decompress and make some waterfall and rank plots if wanted
                #
                if PLOTON:                  
                    for bli in svds:
                        if bli == "meta": continue
                        selbl = bl == bli
                        bla1 = a1[selbl][0]
                        bla2 = a2[selbl][0]
                        plt.figure()
                        plotmarkers = ['xr', '.b', 'dm', '*g'] # up to four supported
                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            V, L, U = svds[bli][corrlbl]['data']
                            plt.plot(L, plotmarkers[ci % len(plotmarkers)], label=corrlbl)
                        plt.legend()
                        plt.yscale("log")
                        plt.xlabel("Singular values")
                        plt.ylabel("Weight")
                        plt.title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                        imname = f"{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                        plt.savefig(os.path.join(OUTPUTFOLDER, imname))
                        plt.close()

                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            corrlbl = ReverseStokesTypes[c]
                            origdata = data[selbl, :, ci].T.copy()
                            selflag = flag[selbl, :, ci].T.copy()
                            origdata[selflag] = np.nan
                            V, L, U = svds[bli][corrlbl]['data']
                            reconstitution = reconstitute(V, L, U, compressionrank=svds[bli][corrlbl]['reduced_rank'])
                            assert reconstitution.shape == origdata.shape
                            reconstitution[selflag] = np.nan
                            
                            fig, axs = plt.subplots(3, figsize=(6,18))
                            imdiff = axs[0].imshow(np.abs(origdata - reconstitution), aspect='auto')
                            cbardiff = fig.colorbar(imdiff, ax=axs[0], orientation='vertical')
                            cbardiff.set_label("Abs difference")
                            axs[0].set_ylabel("Channel")
                            axs[0].set_title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                            imorig = axs[1].imshow(np.abs(origdata), aspect='auto')
                            cbardiff = fig.colorbar(imorig, ax=axs[1], orientation='vertical')
                            cbardiff.set_label("Original data")
                            axs[1].set_ylabel("Channel")
                            imrecon = axs[2].imshow(np.abs(reconstitution), aspect='auto')
                            cbardiff = fig.colorbar(imrecon, ax=axs[2], orientation='vertical')
                            cbardiff.set_label("Reconstructed data")
                            axs[2].set_ylabel("Channel")
                            axs[2].set_xlabel("Sample")
                            imname = f"reconstructed.{corrlbl}.amp.{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                            
                            fig.savefig(os.path.join(OUTPUTFOLDER, imname))
                            plt.close(fig)
                            
                            fig, axs = plt.subplots(3, figsize=(6,18))
                            imdiff = axs[0].imshow(diffangles(np.rad2deg(np.angle(origdata)), 
                                                            np.rad2deg(np.angle(reconstitution))), 
                                                aspect='auto')
                            cbardiff = fig.colorbar(imdiff, ax=axs[0], orientation='vertical')
                            cbardiff.set_label("Phase difference [deg]")
                            axs[0].set_ylabel("Channel")
                            axs[0].set_title(f"Scan {s} bl {antnames[bla1]}&{antnames[bla2]} ({svds[bli]['bllength']:.2f} m)")
                            imorig = axs[1].imshow(np.rad2deg(np.angle(origdata)), aspect='auto')
                            cbardiff = fig.colorbar(imorig, ax=axs[1], orientation='vertical')
                            cbardiff.set_label("Original data [deg]")
                            axs[1].set_ylabel("Channel")
                            imrecon = axs[2].imshow(np.rad2deg(np.angle(reconstitution)), aspect='auto')
                            cbardiff = fig.colorbar(imrecon, ax=axs[2], orientation='vertical')
                            cbardiff.set_label("Reconstructed data [deg]")
                            axs[2].set_ylabel("Channel")
                            axs[2].set_xlabel("Sample")
                            imname = f"reconstructed.{corrlbl}.phase.{VIS}.scan.{s}.bl.{antnames[bla1]}&{antnames[bla2]}.png"
                            
                            fig.savefig(os.path.join(OUTPUTFOLDER, imname))
                            plt.close(fig)
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
                    for bli in svds:
                        if bli == "meta": 
                            p.next()
                            continue
                        selbl = bl == bli
                        bla1 = a1[selbl][0]
                        bla2 = a2[selbl][0]
                        # other ways to do this but we're just going to get the shape from data
                        data = tt.getcol(INPUT_DATACOL)
                        # IMPORTANT:: if correlations are unselected we treat them as a unity operation
                        # uncompressed data is copied to those spots
                        for c in CORRSEL:
                            ci = np.where(corrtypes == c)[0][0]
                            V, L, U = svds[bli][corrlbl]['data']
                            reconstitution = reconstitute(V, L, U, compressionrank=svds[bli][corrlbl]['reduced_rank'])
                            data[selbl,:,ci] = reconstitution.T # we transposed earlier
                        p.next()
                    tt.flush()
                log.info("\t<OK>")
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
    PLOTOVERRIDE=not args.PLOTNOOVERRIDE
    NOSKIPAUTO = args.NOSKIPAUTO
    RANKOVERRIDE = args.RANKOVERRIDE
    SIMULATE = args.SIMULATE
    OUTPUT_DATACOL = args.OUTPUT_DATACOL

    if PLOTON and os.path.exists(OUTPUTFOLDER) and not os.path.isdir(OUTPUTFOLDER):
        raise RuntimeError(f"Output path '{OUTPUTFOLDER}' exists, but is not a folder. Cannot take")
    if PLOTON and PLOTOVERRIDE and os.path.exists(OUTPUTFOLDER):
        shutil.rmtree(OUTPUTFOLDER)
    if PLOTON and not os.path.exists(OUTPUTFOLDER):
        os.mkdir(OUTPUTFOLDER)
    
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
                     OUTPUT_DATACOL)
    