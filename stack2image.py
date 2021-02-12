#!/usr/bin/env python
"""
Convert cutout stack to individual image cutouts.
"""
__author__ = "Alex Drlica-Wagner"
import os
import errno
import logging

import numpy as np
import pylab as plt
import pandas as pd

import fitsio

BANDS  = ['G','R','I','Z']
BINDEX = { b:i for i,b in enumerate(BANDS)}
EXTS   = ['IMAGE','PSF','MASK'] # not carrying the weight...

def mkdir(path):
    """ Create a path an return it 

    Parameters
    ----------
    path : path to create
    
    Returns
    -------
    path : path created
    """
    # https://stackoverflow.com/a/600612/4075339
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    return path

def make_bvalues(amin,ascale):
    """Convert stack scale and min values to FITS BSCALE and BZERO.
    
    Parameters
    ----------
    amin   : array minimum
    ascale : array scale value

    Returns
    -------
    bscale, bzero : FITS header keyword values
    """
    bscale0 = 1.0
    bzero0  = 32768
    amax0   = 65535
    bscale = ascale/amax0 * bscale0
    bzero  = bzero0 * (ascale/amax0) + amin
    return bscale, bzero

def create_header(extname,stack,metadata,index,band):
    """Create a header for an array.

    Parameters
    ----------
    extname  : extension name
    stack    : the fits hdulist of the image stack
    metadata : data frame of metadata
    index    : the index of the image to extract
    band     : the band of the image to extract
    
    Returns
    -------
    hdr      : header dictionary
    """
    if extname == 'IMAGE':
        return create_image_header(stack,metadata,index,band)
    elif extname == 'PSF':
        return create_psf_header(stack,metadata,index,band)
    elif extname == 'MASK':
        return create_mask_header(stack,metadata,index,band)
    else:
        msg = "Unrecognized image type: %s"%extname
        raise ValueError(msg)

def create_image_header(stack,metadata,index,band):
    """Create a header for the image array.

    Parameters
    ----------
    stack    : the fits hdulist of the image stack
    metadata : data frame of metadata
    index    : the index of the image to extract
    band     : the band of the image to extract
    
    Returns
    -------
    hdr      : header dictionary
    """
    index = int(index)
    suffix = '_%s'%band

    hdr = dict()
    hdr['OBJID']  = stack['CUTOUT_ID'][index][0]

    if metadata:
        hdr['RA']     = metadata['RA'][index]
        hdr['DEC']    = metadata['DEC'][index]
         
        for column in metadata.columns:
            if not column.endswith(suffix): continue
            if column.startswith('FILEPATH'): continue
            key = column.rsplit(suffix,1)[0]
            value = metadata[column][index]
            value = np.nan_to_num(value)
            if key == 'CCD_EDGE': value = int(value)
            hdr[key] = value

            path = metadata['FILEPATH_IMAGE'+suffix][index]
            dirname = os.path.dirname(path)
            filename = os.path.basename(path)
            hdr['PATH']  = dirname
            hdr['FILENAME'] = filename

    try:
        # Adjust the WCS
        hdr['CRPIX1'] = hdr['CRPIX1'] - hdr['DCRPIX1']
        hdr['CRPIX2'] = hdr['CRPIX2'] - hdr['DCRPIX2']
    except KeyError as e:
        pass

    if 'IMG_SCALE' in stack:
        # Need to convert to signed 16-bit integers

        minval = np.nan_to_num(stack['IMG_MIN'][index,BINDEX[band]].item())
        scale  = np.nan_to_num(stack['IMG_SCALE'][index,BINDEX[band]].item())
        hdr['IMG_MIN']   = minval
        hdr['IMG_SCALE'] = scale

        bscale, bzero = make_bvalues(minval,scale)
        hdr['BZERO']  = np.nan_to_num(bzero)
        hdr['BSCALE'] = np.nan_to_num(bscale)

    return hdr

def create_psf_header(stack,metadata,index,band):
    """Create a header for the psf array.

    Parameters
    ----------
    stack    : the fits hdulist of the image stack
    metadata : data frame of metadata
    index    : the index of the image to extract
    band     : the band of the image to extract
    
    Returns
    -------
    hdr      : header dictionary
    """
    suffix = '_%s'%band
    hdr = stack['PSF'].read_header()

    if metadata:
        hdr['PSF_SAMP']  = np.nan_to_num(metadata['PSF_SAMP'+suffix][index])
         
        path = metadata['FILEPATH_PSF'+suffix][index]
        dirname  = os.path.dirname(path)
        filename = os.path.basename(path)
        hdr['PATH']  = dirname
        hdr['FILENAME'] = filename

    if 'PSF_SCALE' in stack:
        minval = np.nan_to_num(stack['PSF_MIN'][index,BINDEX[band]].item())
        scale  = np.nan_to_num(stack['PSF_SCALE'][index,BINDEX[band]].item())
        hdr['PSF_MIN']   = minval
        hdr['PSF_SCALE'] = scale

        # Need to convert to signed 16-bit integers
        bscale, bzero = make_bvalues(minval,scale)
        hdr['BZERO']  = np.nan_to_num(bzero)
        hdr['BSCALE'] = np.nan_to_num(bscale)

    return hdr

def create_mask_header(stack,metadata,index,band):
    """ Create header for MASK HDU 

    Parameters
    ----------
    stack    : the fits hdulist of the image stack
    metadata : data frame of metadata
    index    : the index of the image to extract
    band     : the band of the image to extract
    
    Returns
    -------
    hdr      : header dictionary
    """
    hdr = dict()
    return hdr

def parse_index(args,stack):
    """ Convert command line arguments to an array of indices.

    Parameters
    ----------
    args  : command line arguments from argparse
    stack : FITS HDU object

    Returns
    -------
    indexes : array of indexes
    """
    cutids = stack['CUTOUT_ID'].read()
    cutids = cutids[cutids.dtype.names[0]]
    indexes = np.arange(len(cutids))
    
    if args.index:
        index = args.index
        try:
            return indexes[np.in1d(indexes,np.array(index,dtype=int))]
        except:
            index = index[0]
            s = slice(*[int(x) if x else None for x in index.split(':')])
            return indexes[s]
    if args.cutid:
        match = np.in1d(cutids,np.array(args.cutid).astype(cutids.dtype))
        indexes = np.where(match)[0]
        return indexes
    elif args.all:
        return indexes
    else:
        msg = "Missing index argument"
        raise ValueError(msg)

def load_metadata(filename):
    if not filename: return None
    metadata = pd.read_csv(filename)
    # Need to replace NaN in string columns
    columns = [col for col in metadata if col.startswith('FILEPATH')]
    metadata[columns] = metadata[columns].fillna('')

    return metadata

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename',help="cutout file")
    parser.add_argument('metadata',nargs='?',help="metadata file")
    parser.add_argument('-d','--dirname',default='image_cutouts',
                        help="output directory name")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a','--all',action='store_true',
                       help="process all cutouts")
    group.add_argument('-i','--index',nargs='+',type=str,action='append',
                       help="index of cutouts to process (int, list, or slice str)")
    group.add_argument('-c','--cutid',nargs='+',type=str,action='append',
                       help="cutout id to process (int or list)")
    parser.add_argument('-v','--verbose',action='count',
                        help="output verbosity")
    args = parser.parse_args()

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.DEBUG)

    stack = fitsio.FITS(args.filename)
    metadata = load_metadata(args.metadata)

    indexes = parse_index(args,stack)

    dirname = mkdir(args.dirname)

    for index in indexes:
        index = int(index)
        cutid = stack['CUTOUT_ID'][index][0]

        outfile = os.path.join(dirname,'cutout_%s.fits.gz'%cutid)
        if os.path.exists(outfile):
            os.remove(outfile)

        logging.info("Writing %s..."%outfile)
        out = fitsio.FITS(outfile,'rw')
        for extname in EXTS:
            if not extname in stack: continue
            data = stack[extname][index,:,:,:][0]
            for i,band in enumerate(BANDS):
                outname = "%s_%s"%(extname,band)
                hdr = create_header(extname,stack,metadata,index,band)
                out.write(data[i,:,:],extname=outname,header=hdr)
        out.close()
         
