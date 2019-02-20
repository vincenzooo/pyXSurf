"""Similar to points, but manage matrix data with x and y. All functions accept arguments in form data,x,y where x and y can be optional in some routines.

# 2016/10/09 copied from PSDanalysis.py, quite rough, it contains also psd functions that should go in psd2d.

IDL differentiate between functions and procedure. Code synyax is rough (do begin end, -> for methods, comparison operators), but very clear to read. 
For example, allows to understand if modify object (procedure) or
return a result (function). Nobody forbids to a function to modify the argument (e.g. if I modify an object or a variable inside a function, are these changes reflected outside?), however it is usually (always?) not needed, because there is a procedure for that. This also enforces the user to think to the correct interface when the subroutine is called. 
A flag is called as

In python I need to look at code or docstring to understand if a value is modified and this can also not be consistent (some method can work as functions, others as procedures.
self.data=newdata makes a method a procedure, self.copy().data=newdata; return res is a function



"""

import matplotlib.pyplot as plt
import numpy as np
#from pySurf.points import *
#from pySurf.psd2d import *
from pyProfile.profile import polyfit_profile
#from plotting.multiplots import compare_images 
from IPython.display import display
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span, span_from_pixels
from dataIO.outliers import remove_outliers
from dataIO.dicts import strip_kw,pop_kw
import logging
import os
import pdb
from scipy.ndimage import map_coordinates
from scipy import interpolate as ip
from scipy import ndimage
from plotting.captions import legendbox
from astropy.io import fits
from pySurf.testSurfaces import make_prof_legendre, make_surf_legendre


from pySurf.points import points_in_poly, points_autoresample
from plotting.add_clickable_markers import add_clickable_markers2
from pySurf.find_internal_rectangle import find_internal_rectangle
import itertools

## SEMI-OBSOLETE LEVELING WITH 1D AUXILIARY FUNCTIONS
## STILL ONLY WAY TO HANDLE GENERIC (non-legendre) leveling by line, 
##   code needs review for the part where the leveling function needs x,
##   see example 
"""This is the old leveling. level_by_line """
    
def removelegendre(x,deg):
    """remove degree polyomial, a possible leveling function for leveldata.
    Note: this is superseded by levellegendre"""
    print ("removelegendre is obsolete, use levellegendre instead.")
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval
    def rem(y):
        return y-lval(x,lfit(x,y,deg),tensor=0)   
    return rem
  
def removesag(y):
    """remove second degree polyomial, a possible leveling functiondata for leveldata."""
    return y-polyfit_profile (y,degree=2)

def removept(y):
    """remove piston and tilt, a possible leveling functiondata for leveldata."""
    #never tested
    y=y-line (y)
    return y-np.nanmean(y)
    
def level_by_line(data,function=None,axis=0,**kwargs):
    """remove line through extremes line by line (along vertical lines).
    The returned array has 0 at the ends, but not necessarily zero mean.
    If fignum is set, plot comparison in corresponding figure.
    Function is a function of profile vector y that returns a corrected profile.
    
    Completely useless, can be replaced by np.apply_along_axis"""
    
    print ("level_by_line is completely useless function, use np.apply_along_axis.")

    def remove_line(y): #default line removal function
        return y-line(y) 
    
    if function is None:
        function=remove_line
        
    ldata=np.apply_along_axis( function, axis=axis, arr=data, **kwargs )
        
    return ldata

#more updated leveling functions

def fitlegendre(x,y,deg,nanstrict=False,fixnans=False):
    """Return a legendre fit of degree deg. Work with 1 or 2D y (if 2D, each column is independently fit
    and x is the coordinate of first axis.
    if nanstrict is True, every column containing nan (or Inf) is considered invalid and a column of nan is returned, if False, nan are excluded and fit is calculated on valid points only (note that since columns
    are slices along first index, the option has no effect on 1D data (nans are always returned as nans)."""
    
    '''
    note this was working with nan 
    def removelegendre(x,deg):
    """remove degree polyomial, a possible leveling function for leveldata."""
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval
    def rem(y):
        return y-lval(x,lfit(x,y,deg),tensor=0)   
    return rem
    '''
    
    #this works for 1D or 2D and fits 
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval

    import pdb
    #pdb.set_trace()
            
    ## goodind definition can be adjusted for > 2D , datarec definition is already good thanks to [...,] slice
    goodind=np.where(np.sum(~np.isfinite(y),axis=0)==0)[0] #indices of columns non containing any nan
        
    #fit legendre of degree. lval transpose data (!) so it must be transposed back if 2D
    result=y*np.nan
    if len(goodind)>0:
        #this applies only to 2D data
        datarec=lval(x,lfit(x,y[...,goodind],deg))
        if len(y.shape)==2:
            datarec=datarec.T    
        result[...,goodind]= datarec
    
    if len(y.shape)==2:
        if not nanstrict:
            nancols=np.isin(np.arange(y.shape[-1]),goodind,invert=True) #boolean
            if nancols.any():
                datarec=np.zeros((y.shape[0],y.shape[1]-len(goodind)))*np.nan
                for i,col in enumerate(y[...,nancols].T):
                    datarec[:,i]=fitlegendre(x,col,deg,fixnans=fixnans)
                result[:,nancols]=datarec
            # recursively fit columns containing nans one by one to filter nans
            """
            nancols=np.isin(np.arange(y.shape[-1]),goodind,invert=True) #boolean 
            datarec=np.zeros(len(x)*np.nan
            datarec=datarec.T 
            for i,(xx,col) in enumerate(zip(x[nancols],y[...,nancols])):
                datarec[]
            """
    elif len(y.shape)==1:
        mask=np.isfinite(y)
        if np.logical_not(mask).all():
            result=y
        else:
            #if nan need to be fixed coefficients are used on all x,
            # if not fixed only good points are replaced
            coeff=lfit(x[mask],y[mask],deg)
            if fixnans:
                result=lval(x,coeff)
            else:
                result=y*np.nan
                result[mask]=lval(x[mask],coeff)
        
    return result

def plot_stats(datalist,x=None,y=None,bins=100,labels=None,*args,**kwargs):
    """plot histogram and returns statistics. Experimental.
    Ignore invalid values. Data are not leveled.
    Returns array of rms.
    Additional keywords are passed to plt.hist (might fail for duplicate keywords)
    
    usage:
    plt.figure('stats')
    plt.clf()
    stats = plot_stats ([wdata,wdata2,ddata],labels=['Original',
    'Figured','Difference'])
    plt.title('Height Distribution')  #change title from default
    
    """
    
    plt.figure('hist')
    plt.clf()
    plt.title('Height Distribution')
    
    if np.array(datalist).size == 1:
        datalist = [datalist]
    if labels is None:
        labels=[""]*len(datalist)
    rms=[]
    for d,l in zip(datalist,labels):
        plt.hist(d[np.isfinite(d)],bins=bins,label=l+', rms=%.3g $\mu$m'%np.nanstd(d),alpha=0.2
            ,*args,**kwargs)
        rms.append(np.nanstd(d))
    plt.legend(loc=0)
    return rms
    
def levellegendre(x,y,deg,nanstrict=False):
    """remove degree polyomial by line, evolution of leveldata using legendre functions
    that work also 2D. nr. of terms fitted in legendre is deg+1 (piston->deg=0).
    For 2D data, data are passed as second argument (y) and y coordinates passed as first (x) (legendre are leveled along columns). Nanstrict excludes the lines containing nans, otherwise only good points are considered."""
    
    datarec=fitlegendre(x,y,deg,nanstrict=nanstrict)

    #result[...,goodind]= y[...,goodind]-datarec #remove poly by line 
    result=y-datarec
    return result #y-datarec 

def level_data(data,x=None,y=None,degree=(1,1),byline=False,fit=False,*arg,**kwarg):
    """use RA routines to remove degree 2D legendres or levellegendre if leveling by line. 
    Degree can be scalar (it is duplicated) or 2-dim vector. must be scalar if leveling by line.
    leveling by line also hondle nans.
    x is not used, but maintained for interface consistency.
    fit=True returns fit component instead of residuals"""
    from utilities.imaging.fitting import legendre2d
    
    if x is None:  x = np.arange(data.shape[1])
    if y is None:  y = np.arange(data.shape[0])
    if byline:
        if np.size(degree)!=1:
            raise ValueError("for `byline` leveling degree must be scalar: %s"%degree)
        leg=fitlegendre(y,data,degree,*arg,**kwarg)  #levellegendre(x, y, deg, nanstrict=False)
        
        #pdb.set_trace()
    else:    
        if np.size(degree)==1: degree=np.repeat(degree,2)    
        leg=legendre2d(data,degree[0],degree[1],*arg,**kwarg)[0] #legendre2d(d, xo=2, yo=2, xl=None, yl=None)
    return (leg if fit else data-leg),x,y
    
## 2D FUNCTIONS

def transpose_data(data,x,y):
    """Transpose data and coordinates, return new data,x,y.
    
    return a view, see np.ndarray.T and np.ndarray.transpose for details"""
    
    data=data.T
    x,y=y,x
    return data,x,y
    
def apply_transform(data,x,y,trans=None):
    """applies a 3D transformation (from Nx3 to Nx3) to data. 
    TODO: add option to set final resampling grid keeping initial sampling, initial number of points or on custom grid (use points.resample_grid, resample_data)."""
    from pySurf.points import matrix_to_points2,points_autoresample
    
    if trans is not None:
        p2=trans(matrix_to_points2(data,x,y))
        data,x,y=points_autoresample(p2)
    
    return data,x,y
    
    
def rotate_data(data,x=None,y=None,ang=0,k=None,center=None,
    fill_value=np.nan,usepoints=False,*args,**kwargs):
    """Rotate clockwise by an angle in degree. Non optimized version using points functions.    
    2018/10/31 added k parameters allowing 90 deg rotations with np.rot90. k is the number
        of anticlockwise rotations about center. Note there is not resampling, so it can be inaccurate if
        center is not on pixel center.
    
    rotate_Data was intended to work using scipy.ndimage.interpolation.rotate but this failed.
    Added 2018/12/12
    args and kwargs are passed to the function that handles the rotation.
    
    See also comments on resampling in apply_transform.
    
    rot90 determination of rotated axis can probably be extended to general case, but in the meanwhile
    the implementation based on points offers an accurate interpolation (even if slower),
      can be enabled setting flag `usepoints`.
    
    """
    
    #establish data coordinates if not provided
    if x is None: 
        x=np.arange(data.shape[1],dtype=float)
    if y is None:
        y=np.arange(data.shape[0],dtype=float)

    corners=list(itertools.product((x[0],x[-1]),(y[0],y[-1])))    
    xc,yc=span(corners,axis=0).mean(axis=0)  #center of data in data coordinates
    if center is None: #rotation center
        center=(xc,yc)  
    step=x[1]-x[0],y[1]-y[0] #this is kept constant with rotation
    
    if usepoints:
        #doesn't work well for some reason (nan?)
        from pySurf.points import matrix_to_points2,rotate_points,resample_grid
        p=matrix_to_points2(data,x,y)
        p=rotate_points(p,-ang/180*np.pi,center=center,*args,**kwargs)
        #res=resample_grid(p,x,y,matrix=True),x,y  #this is wrong, needs to rescale on grid on rotated range
        #similar to data rotation, for now use autoresample
        res=points_autoresample(p)
        #raise NotImplementedError
        return res
        
    if k is not None:
        assert ang is None
        
        ##alg 1: basic rotation with maximum inplace operations, messes up sign of axis, raises ambiguity on how to plot with plot_data
        #data=np.rot90(data,k,axes=(1,0))
        #return data,-y,x        #note that after this: y is same as original x, while y is a whole new variable.
        #                        #data is instead a view (change of elements affects also original data).

        #alg 3: give up inplace operations for axis, center is implemented and determine axis, 
        # data are copied on return, without it (alg 2), rotated data would be still a view of the original data.            
        data=np.rot90(data,k,axes=(1,0),*args,**kwargs)        
        for i in range(k):            
            xlim,ylim=span(x),span(y)
            x2=center[0]+center[1]-y
            y2=center[1]-center[0]+x
            x2.sort()
            y2.sort()
            x,y=x2,y2
        return data.copy(),x,y        
    
        #note that after the following: y is same as original x, while y is a whole new variable.
        #data is instead a view (change of elements reflects to original variable).
        #data=np.rot90(data,k,axes=(1,0)).copy()
        #x,y=-y,x
        #return data,x,y
      
    # use ndimage to rotate data and recalculate coordinate 
    #this assumes same center and non-integer coordinates are interpolated on integer cordinates 
    #   (in units of steps) 
    # default cval = 0.0 
    a3=ndimage.rotate(data,ang,reshape=1,cval=fill_value,*args,**kwargs)
    x3=np.arange(a3.shape[1],dtype=float)*step[0]
    y3=np.arange(a3.shape[0],dtype=float)*step[1]

    #this is center of integer indices of rotated data, different from float center
    #  of rotated corner span (can be calculated with matrix, see ipynb notebook
    corners3=list(itertools.product((x3[0],x3[-1]),(y3[0],y3[-1])))
    xc3,yc3=span(corners3,axis=0).mean(axis=0)

    th=ang/180*np.pi
    #rotate distance vector of rotation center from scipy center 
    x0,y0=np.matmul(([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]]),(np.array([xc,yc])-center)).T
    #x0,y0=rotate_points((np.array([xc,yc])-center),ang*np.pi/180.,center=center) #rotation center transformed using points
    
    #x0+xc is center of rotated corners
    xout=x3-(xc3)+x0+center[0]  #(x3-xc3)*step[0]+center[0]
    yout=y3-(yc3)+y0+center[1] #(y3-yc3)*step[1]+center[1]
    
    
    return a3,xout,yout

    

def save_data(filename,data,x=None,y=None,fill_value=np.nan,addaxis=True,makedirs=False,**kwargs):
    """save data as matrix on a file. 
    Can save as fits if the extension is .fits, but this should be probably moved elsewhere,
    otherwise uses np.saavetxt to save as text.
    kwargs are passed to np.savetxt or hdu.writeto"""
    #2016/10/21 rewritten routine to use points_find_grid
    #2014/08/08 default fill_value modified to nan.
    #20140420 moved filename to second argument for consistency.

    #fix x and y according to input
    #calculate x and y from shape if provided, then overwrite if other options 
    shape=data.shape
    if x is None:
        x=np.linspace(0,shape[1],shape[1])
    if y is None:
        y=np.linspace(0,shape[0],shape[0])
    #x and y are automatically calculated, then overwrite if xgrid and ygrid provided
    
    if makedirs:
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        
    #save output
    if os.path.splitext(filename)[-1]=='.fits':
        print("creating fits..")
        hdu=fits.PrimaryHDU(data)
        units=kwargs.pop('units','')
        if np.size(units)==1:
            units=np.repeat(units,3)        
        
        hdu.writeto(filename,overwrite=1,**kwargs)    
        fits.setval(filename,'XPIXSZ',value=span(x,size=True)/(np.size(x)-1),comment='Image pixel X size'+ ("" if units is None else (' in '+ units[0])) )
        fits.setval(filename,'XPIYSZ',value=span(y,size=True)/(np.size(y)-1),comment='Image pixel Y size'+ ("" if units is None else (' in '+ units[1])) )
        return
    else:
        
        if addaxis:
            #add first column and row with x and y coordinates, unless flag addaxis is set false
            data=np.vstack([x,data])
            data=np.hstack([np.concatenate([[np.nan],y])[:,None],data])
    
    np.savetxt(filename,data,**kwargs)

def register_data(data,x,y,crop=None,center=None,
    strip=False,scale=(1,1,1.),*args,**kwargs):
    """Read a data matrix file in format accepted by a reader with appropriated options. Returns data,x,y.
    
    Can crop data and translate on three coordinates. crop is calculated on raw data,
    center is the position of the central point after crop and scaling.
    If strip is set, nan are stripped and center is calculated on valid data only.
    Scale is applies to data after format-specific options, but before centering. 
    
    It can be used e.g. to invert x axis as in measuring 220 mm mandrel with 1 m CGH (scale=(-1,1,1)).    
    
    Each reader is a function that handles a set of format-specific parameters. All paramenters that can be derived from file header must be handled by the reader. 
    This is pretty much the only limitation, any function with at least one parameter
        (typically filename) returning a triple data,x,y can be used as reader function.
    If reader is not provided, a guess is attempted.
    reader-specific parameters:
    load4D:
        ypix is the size of pixel in vertical direction.
        ytox the conversion factor (typically given by radius of test optics/radius of CGH).
        
    *args and **kwargs are not used, just put here to suppress error if additional parameters are passed.

"""
    
    x=x*scale[0]
    y=y*scale[1]
    data=data*scale[2]   #*zscale
    #if any x or y is inverted invert data and orient as cartesian.
    #this maybe can be move somewhere else, maybe in class set_data,x,y
    # but be careful to avoid double reflection.    
    if x[-1]<x[0]:
        x=x[::-1]
        data=np.fliplr(data)
        x=x-min(x)
    if y[-1]<y[0]:
        y=y[::-1]
        data=np.flipud(data)  
        y=y-min(y)        
    #x=x-min(x) this should have effect only if axis had been inverted, so I moved them inside the conditional  2018/04/19
    #y=y-min(y) 

    #adjust crop and scales
    if strip:  #messes up x and y 
        #print("WARNING: strip nans is not properly implemented and messes up X and Y.")
        data,x,y = remove_nan_frame(data,x,y)
    
    #doing cropping here has the effect of cropping on cartesian orientation,
    #coordinates for crop are independent on center and dependent on scale and pixel size.
    #center is doing later, resulting in center of cropped data only is placed in the suitable position.
    if crop is None:
        crop=[None,None,None]
    data,x,y=crop_data(data,x,y,*crop)
    
    #center data if center is None, leave unchanged
    if center is not None:
        assert len(center)>=2
        x=x-(np.max(x)+np.min(x))/2.+center[0]
        y=y-(np.max(y)+np.min(y))/2.+center[1]
        if len(center)==3:
            data=data-(np.nanmax(data)+np.nanmin(data))/2.+center[2]    
    
    return data,x,y

def data_equal(d1,d2,nanstrict=False):
    """Compare two data keeping nans into account."""
    raise NotImplementedError
    
def read_data(file,reader,*args,**kwargs):
    """non essendo sicuro dell'interfaccia per ora faccio cosi'.
    The function first calls the data reader, then applies the register_data function to address changes of scale etc.
        This works well, however read_data must filter the keywords for the reader and for the register and
        this is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is 
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for 
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers, 
        while this can be done using read_data. """
    
    """this is an ugly way to deal with the fact that return
    arguments are different if header is set, so when assigned to a variable as in patch routines in pySurf instrumentReader it fails.
    Workaround has been calling directly read_data, not optimal."""
    if kwargs.pop('header',False):
        try:
            return reader(file,header=True)
        except TypeError:  #unexpected keyword if header is not implemented
            return None
            
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    #zscale=kwargs.pop('zscale',None) this is removed and used only for reader
    # functions where a conversion is needed (e.g. wavelength)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    
    data,x,y=reader(file,*args,**kwargs)
    return register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)

            
# this is old version before read_data. Note that x and y have no effect, has a autocrop parameter that removes nans.
# pass options to np.genfromtxt
def get_data(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,addaxis=False,center=None,skip_header=None,delimiter=' ',autocrop=False):
    """replaced by data_from_txt."""
    print ('this routine was replaced by `data_from_txt`, update code')
 
def data_from_txt(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,addaxis=False,center=None,skip_header=None,delimiter=' ',autocrop=False):
    """read matrix from text file. Return data,x,y
    center is the position of the center of the image in final coordinates (changed on 2016/08/10, it was '(before any scaling or rotation) in absolute coordinates.') If None coordinates are left unchanged.
        Set to (0,0) to center the coordinate system to the data.
    addaxis (if matrix is set) can be set to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_data.
    autocrop remove frame of nans (external rows and columns made all of nans),
        note it is done before centering. To include all data in centering, crop data
        at a second time with remove_nan_frame
    """

    #2014/04/29 added x and y as preferred arguments to xrange and yrange (to be removed).
    if skip_header is None: 
        skip=0 
    else: 
        skip=skip_header
    
    mdata=np.genfromtxt(filename,skip_header=skip,delimiter=delimiter)
    if addaxis:
        y,mdata=np.hsplit(mdata,[1])
        y=y[1:].flatten() #discard first corner value
        x,mdata=np.vsplit(mdata,[1])
        x=x.flatten() #remove first dimension
    else:
        x=np.arange(mdata.shape[1])
        y=np.arange(mdata.shape[0])

    if autocrop:
        mdata,x,y=remove_nan_frame(mdata,x,y)
        
    if center is not None:
        assert len(center)==2
        x=x-(((np.nanmax(x)+np.nanmin(x))/2)-center[0])
        y=y-(((np.nanmax(y)+np.nanmin(y))/2)-center[0])
    
    return mdata,x,y


def resample_data(d1,d2,method='mc',onfirst=False):  

    """resample d1 [Ny' x Nx'] on x and y from d2[Nx x Ny], where d1 and d2
    are passed as list of data,x,y.
    Return a [Nx x Ny] data. 
    onfirst allow to resample second array on first (same as swapping args).
    To get a (plottable) matrix of data use:
    plt.imshow(rpoints[:,2].reshape(ygrid.size,xgrid.size))."""

    from scipy.interpolate import interp1d
    
    if onfirst:
        d1,d2=d2,d1
    
    try:
        data1,x1,y1=d1
    except ValueError:
        data1=d1
        x1,y1=np.arange(d1.shape[1]),np.arange(d1.shape[0])
        
    try:
        data2,x2,y2=d2 #grid for resampling. data is useless.
    except ValueError:
    #x and y not defined, use data1. 
        x2,y2=np.linspace(*span(x1),d2.shape[1]),np.linspace(*span(y1),d2.shape[0])
    
    try:
        #if data_equal((data1,x1,y1),(data2,x2,y2)):
        # x and y should not contain nan, but just in case do full comparison to avoid
        # nan != nan
        #raise ValueError
        if ((x1 == x2) | (np.isnan(x1) & np.isnan(x2))).all() & ((y1 == y2) | (np.isnan(y1) & np.isnan(y2))).all():
            return data1,x1,y1
    except ValueError:
        pass #elements have different shape, they are different and need resampling
    
    #data2,x2,y2=d2 
    if method=='gd':   #use griddata from scipy.interpolate     
        z=ip.griddata(np.array(np.meshgrid(x1,y1)).reshape(2,-1).T,data1.flatten(),np.array(np.meshgrid(x2,y2)).reshape(2,-1).T) #there must be an easier way
        z.reshape(-1,len(x2))
        
        #old not working z=ip.griddata(np.meshgrid(x1,y1),data1,(x2,y2),method='linear') #this is super slow, 
    elif method == 'mc':  #map_coordinates
        def indint (x1,x2):
            """interpolate x2 on x1 returning interpolated index value.
            Es.: x2=[1.5,5.5,9.5]; x1=[1.,2.,9.,10.]  --> [0.5,1.5,2.5]"""
            #replaced with scipy.interpolate.interp1d that allows extrapolation, this guarantees that output has correct number of points even if ranges are not contained one into the other (they are "intersecting").
            f=interp1d(x2,np.arange(len(x2)),fill_value='extrapolate')
            i=f(x1)            #i=np.interp(x1,x2,np.arange(len(x2)),left=np.nan,right=np.nan)
            #return i[np.isfinite(i)]
            return i
            
        xind=indint(x2,x1)
        yind=indint(y2,y1)    
        z=map_coordinates(data1,np.meshgrid(xind,yind)[::-1],cval=np.nan,order=1)
        
    return z,x2,y2

def subtract_data(d1,d2,xysecond=False,resample=True):
    """d1 and d2 are triplets (data,x,y), second array is automatically resampled
    on first, unless resample is set to False (in that case, data arrays are assumed
    having same size and subtracted.
    If xySecond is set to True results are calculated on xy of 2nd array,
    equivalent to -(subtract_data(d2-d1))."""
        
    #if resample: d2=resample_data(d2,d1,onfirst=not(xysecond)) #if xysecond resample on d2, but still subtract d2 from d1.
    
    if xysecond:
        res = subtract_data(d2,d1)
        return -res[0],res[1],res[2]
    else:
        if resample: d2=resample_data(d2,d1)
        
        return d1[0]-d2[0],d1[1],d1[2]   
        #note that this returns components of original array for x and y
    
def sum_data(d1,d2,xysecond=False,resample=True):
    """Sum two data sets after interpolation on first set coordinates.
    If xySecond is set to True results are calculated on xy of 2nd array."""
        
    if resample: d2=resample_data(d2,d1,onfirst=xysecond)
  
    return d1[0]+d2[0],d1[1],d1[2]  
    


def grid_in_poly(x,y,verts):
    """data is not needed. Example of usage?"""
    
    p=np.array([xy.flatten() for xy in np.array(np.meshgrid(x,y))]).T
    return np.reshape(points_in_poly(p,verts),(len(y),len(x)))
    
def crop_data(data,x,y,xrange=None,yrange=None,zrange=None,mask=False,poly=None,
        interactive=False,*args,**kwargs):
    """return data,x,y """
    
    outmask=np.ones(np.shape(data),dtype=bool)
        
    if interactive:
        """minimally tested."""

        curfig=plt.gcf() if plt.get_fignums() else None
        fig=plt.figure()
        plot_data(data,x,y,*args,**kwargs)        
        print ("""Zoom to the region to crop, and/or use CTRL+leftClick to add points and create
        an polygonal selection. CTRL+rightClick remove the nearest point. Press ENTER when done.""")
        ax=add_clickable_markers2(hold=True,propertyname='polypoints')
        if xrange is None:
            xrange=plt.xlim()
        if yrange is None:
            yrange=plt.ylim()
        mh=ax.polypoints
        del ax.polypoints
        poly=mh
        plt.close(fig)
        if curfig:
            plt.figure(curfig.number);
            
    #set ranges handling possible Nones in extremes
    xrange=span(x) if (xrange is None) else \
        np.where([xx is None for xx in xrange],span(x),xrange)
    yrange=span(y) if (yrange is None) else \
        np.where([xx is None for xx in yrange],span(y),yrange)
    zrange=span(data) if zrange is None else \
        np.where([xx is None for xx in zrange],span(z),zrange)
        
    import pdb
    #spdb.set_trace()
    if poly:
        outmask=outmask & grid_in_poly(x,y,poly)
        
    if mask:
        outmask [:,(x<xrange[0]) | (x>xrange[1])] = False
        outmask [(y<yrange[0]) | (y>yrange[1]),:] = False   
        outmask [((data>zrange[1]) | (data<zrange[0]))] = False
        res = outmask,x,y
    else:
        #could be made by setting nans and removing frame, but it would 
        #  remove also extra nan lines, so this is the safer way,
        # but it doesn't crop, so you need to remove frame manually on return
        # if you want.
        
        data[~outmask]=np.nan
        data=data[:,(x>=xrange[0]) & (x<=xrange[1])]
        data=data[(y>=yrange[0]) & (y<=yrange[1]),:]
        x=x[(x>=xrange[0]) & (x<=xrange[1])]
        y=y[(y>=yrange[0]) & (y<=yrange[1])]
        data[((data>zrange[1]) | (data<zrange[0]))]=np.nan 
        res=data,x,y
        
    return res

    
def crop_data0(data,x,y,xrange=None,yrange=None,zrange=None):
    
    """original version before adding polygonal and interactive selection."""
    #set ranges handling possible Nones in extremes
    if xrange is None: 
        xrange=span(x)
    else:
        xrange = np.where([xx is None for xx in xrange],span(x),xrange)
    if yrange is None: 
        yrange=span(y)
    else:
        yrange = np.where([xx is None for xx in yrange],span(y),yrange)
    if zrange is None: 
        zrange=span(data)
    else:
        zrange = np.where([xx is None for xx in zrange],span(z),zrange)
    import pdb
    
    #spdb.set_trace()
    data=data[:,(x>=xrange[0]) & (x<=xrange[1])]
    data=data[(y>=yrange[0]) & (y<=yrange[1]),:]
    x=x[(x>=xrange[0]) & (x<=xrange[1])]
    y=y[(y>=yrange[0]) & (y<=yrange[1])]
    data[((data>zrange[1]) | (data<zrange[0]))]=np.nan 
    return data,x,y
    
def remove_nan_frame(data,x,y,internal=False):
    """remove all external rows and columns that contains only nans. If internal is set, return the
    internal crop (largest internal rectangle without nans on the frame)."""
    
    #resuls are obtained by slicing, 
    nancols=np.where(np.sum(~np.isnan(data),axis=0)==0)[0] #indices of columns containing all nan
    if len(nancols)>0:
        try:
            istart=nancols[nancols==np.arange(len(nancols))][-1]+1
        except IndexError:
            istart=0
        try:
            istop=nancols[np.arange(data.shape[1])[-len(nancols):]==nancols][0]
        except IndexError:
            istop=data.shape[1]+1
        data=data[:,istart:istop]
        x=x[istart:istop]
    
    #repeat for rows
    nanrows=np.where(np.sum(~np.isnan(data),axis=1)==0)[0] 
    if len(nanrows)>0:
        #indices of columns containing nan
        try:
            istart=nanrows[nanrows==np.arange(len(nanrows))][-1]+1
        except IndexError:
            istart=0            
        try:
            istop=nanrows[np.arange(data.shape[0])[-len(nanrows):]==nanrows][0]
        except IndexError:
            istop=data.shape[0]+1
        data=data[istart:istop,:]
        y=y[istart:istop]    
    
    if internal:
        r=find_internal_rectangle(data,x,y)
        data,x,y=crop_data(data,x,y,*r)
    
    return data,x,y
    
def projection(data,axis=0,span=False,expand=False):

    """return average along axis. default axis is 0, profile along x.
    keywords give extended results:
    span: if set, return  3 vectors [avg, min, max] with min and max calculated pointwise along same direction.
    expand: instead of a single vector with point-wise minima, returns lists of all vectors having at least one point 
        that is minimum (maximum) between all vectors parallel to axis. Overrides span.
    ex:
    a=array([[21, 16,  3, 14],
             [22, 17,  6, 15],
             [ 0,  3, 21, 16]])
       
    In [62]: projection(a)
    Out[62]: array([ 14.33333333,12.,10.,15.])
    
    In [73]: projection(a,span=True)
    Out[73]:
    [array([ 14.33333333,12.,10.,15.]),
     array([ 0,  3,  3, 14]),
     array([22, 17, 21, 16])]
    
    In [71]: projection(a,expand=True)
    Out[71]:
    [array([ 14.33333333,12.,10.,15.]),
     array([[21, 16,  3, 14],[ 0,  3, 21, 16]]),
     array([[22, 17,  6, 15],[ 0,  3, 21, 16]])]
    """
    
    pm=np.nanmean(data,axis=axis) 
    
    if expand:
        imin=np.sort(list(set(np.nanargmin(data,axis=axis))))
        pmin=np.take(data,imin,axis)
        imax=np.sort(list(set(np.nanargmax(data,axis=axis))))
        pmax=np.take(data,imax,axis)
        pm=[pm,pmin,pmax]
    elif span:
        pmin=np.nanmin(data,axis=axis) 
        pmax=np.nanmax(data,axis=axis) 
        pm=[pm,pmin,pmax]
    
    return pm
    
def matrix_project(data,axis=1):

    """project a matrix along an axis and return, min, mean and max. 
    For backward compatibility, replaced by projection
    (note different default axis)."""
    
    #return np.nanmin(data,axis=axis),np.nanmean(data,axis=axis),np.nanmax(data,axis=axis)
    return projection(data,span=True,axis=axis)

    
def calculate_slope_2D(wdata,x,y,scale=(1.,1.,1.)):
    """calculate slope maps in x and y.
    return slope in x and y respectively.
    Set scale to (dx,dy,1000.) for z in micron, x,y in mm."""    
    
    dx=np.diff(x)#[:,np.newaxis]
    dy=np.diff(y) [:,np.newaxis]  #original
     
    grad=np.gradient(wdata)  #0->axial(along y), 1->az (along x)
    slopeax=grad[0][:-1,:]/scale[-1]*206265/dy  #in arcseconds from um  #original
    slopeaz=grad[1][:,:-1]/scale[-1]*206265/dx  
    #slopeax=grad[0][:,:-1]/scale[-1]*206265/dy  #in arcseconds from um
    #slopeaz=grad[1][:-1,:]/scale[-1]*206265/dx    
    
    return slopeax,slopeaz

def get_stats(wdata,x=None,y=None,units=None):
    """"""
    if units is None:
        u = ""     
    elif isinstance(units,str):
        u=units
    else:
        u=units[-1] if units[-1] is not None else ""
    stats=['RMS:%3.3g %s'%(np.nanstd(wdata),u),'PV:%3.3g %s'%(span(wdata,size=1),u)]
    return stats

    
def data_histostats(data,x=None,y=None,bins=100,normed=True,units=None,loc=0,*args,**kwargs):
    """wrapper around plt.hist, plot histogram of data (over existing window) adding label with stats."""
    units=units if units is not None else ['[X]','[Y]','[Z]']
    res = plt.hist(data[np.isfinite(data)],bins=bins,normed=normed,*args,**kwargs)
    plt.xlabel('Units of '+units[2])
    plt.ylabel('Fraction of points #')
    legend=get_stats(data,x,y,units=units,level=2)+"\n"+'x_span: %.3g\ny_span: %.3g\nN: %i'%(span(x,size=1),
        span(y,size=1),np.size(data))
    l=legendbox(legend,loc=loc,framealpha=0.2)
    #plt.legend(loc=0)
    return res
    
    
## PLOT FUNCTIONS

def plot_data(data,x=None,y=None,title=None,outfile=None,units=None,stats=False,loc=0,
    largs=None,framealpha=0.5,nsigma=None,*args,**kwargs):
    """Plot data using imshow and modifying some default properties.
    Units for x,y,z can be passed as 3-el array or scalar, None can be used to ignore unit.
    Broadcast all imshow arguments.
    stats is a flag to print stats on plot. Here it can be assigned booleanTrue
    or integer value: True or 1 plots statistics, 2 plots x and y spans
    largs is a dictionary of arguments passed to caption in plotting.caption.legendbox 
    (used only if stats are plotted).
        (e.g. {'color':'r'}
    Returns axim as returned by plt.imshow.
    nsigma set colorscale to this multiple of data standard deviation. If dict, a nummber of parameters 
    for outliers removal can be passed to determine color range (data are left intact)."""
    
    aspect=kwargs.pop('aspect','equal')
    origin=kwargs.pop('origin',"lower")
    interpolation=kwargs.pop('interpolation',"none")
    if largs is None:
        largs={}
    
    """aspect,origin,interpolation = pop_kw(
                    kwargs, 
                    {'aspect':'equal',
                    'origin':"lower",
                    'interpolation',"none"},
                    ['aspect','origin','interpolation'])"""
    
    if np.size(units)==1:
        units=np.repeat(units,3)
    
    if x is None: x = np.arange(data.shape[1])
    if y is None: y = np.arange(data.shape[0])
    assert x[-1]>x[0] and y[-1]>y[0]
    
    
    sx=span_from_pixels(x)
    sy=span_from_pixels(y)
    axim=plt.imshow(data,extent=(sx[0],sx[1],sy[0],sy[1]),
        aspect=aspect,origin=origin,interpolation='none',**kwargs)
    #axim=plt.imshow(data,extent=(span(x)[0]-dx,span(x)[1]+dx,span(y)[0]-dy,span(y)[1]+dy),
    #    **pop_kw(kwargs,{'aspect':'equal',
    #                'origin':"lower",
    #                'interpolation':"none"}))
    
    if nsigma is not None:
        if isinstance(nsigma,dict):
            clim=remove_outliers(data,span=True,**nsigma)
        else:
            clim=remove_outliers(data,span=True,nsigma=nsigma)
        plt.clim(*clim)
        print('clim',clim)
        
    plt.xlabel('X'+(" ("+units[0]+")" if units[0] is not None else ""))
    plt.ylabel('Y'+(" ("+units[1]+")" if units[1] is not None else ""))
    
    #im_ratio = data.shape[0]/data.shape[1]   #imperfect solution found on stack overflow. Doesn't work in some cases (3 panels? logaritmic axis?)
    ax=plt.gca()  #try my variation
    s=ax.get_position().size
    #im_ratio = data.shape[0]/data.shape[1]   #imperfect solution 
    im_ratio = s[0]/s[1] # attempt improvement
    cb = plt.colorbar(axim,fraction=0.046*im_ratio, pad=0.04)
    #cb=plt.colorbar() 
    if units[2] is not None:
        cb.ax.set_title(units[2])
        #cb.ax.set_xlabel(units[2],rotation=0)
        #cb.ax.xaxis.set_label_position('top')
        #cb.ax.xaxis.label.set_verticalalignment('top') #leave a little bit more room
    if stats:
        legend=get_stats(data,x,y,units=units)
        if stats==2:
            legend.extend(["x_span: %.3g %s"%(span(x,size=1),
            units[0]),"y_span: %.3g %s"%(span(y,size=1),units[1]),
            "size: %i"%np.size(data)])
        l=legendbox(legend,loc=loc,framealpha=framealpha)
        #pdb.debug()
        #for k,v in largs.items():
            #for text in l.get_texts():
                #text.set_color("red")
        #plt.setp(l.get_texts(), color='r')
    if title is not None:
        plt.title(title)
    if outfile is not None:
        plt.savefig(outfile)
    return axim

def xplot_data(data,x,y,title=None,outfile=None,*args,**kwargs):
    aspect=kwargs.pop('aspect','equal')
    origin=kwargs.pop('origin',"lower")
    axim=plt.imshow(data,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect=aspect,origin=origin,**kwargs)
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if outfile is not None:
        plt.savefig(outfile)
    return axim
"""
this fails in giving plots with better scaled axis.
fig=plt.figure()
ax1=fig.add_subplot(311, adjustable='box-forced')
ax1.imshow(np.random.rand(100).reshape((10,10)),extent=(0,20,0,20),aspect='equal')
ax2=fig.add_subplot(312,sharex=ax1, adjustable='box-forced')
ax2.imshow(np.random.rand(100).reshape((10,10)),extent=(0,20,0,200),aspect='auto')
"""

def mark_data(datalist,outfile=None,deg=1,levelfunc=None,propertyname='markers',direction=0):
    """plot all data in a set of subplots. Allows to interactively put markers 
    on each of them. Return list of axis with attached property 'markers'"""
    import logging
    from matplotlib.widgets import MultiCursor
    from plotting.add_clickable_markers import add_clickable_markers2
    
    try: 
        logger
    except NameError:
        logger=logging.getLogger()
        logger.setLevel(logging.DEBUG)
    
    axes=list(compare_images([[levellegendre(y,wdata,deg=6),x,y] for (wdata,x,y) in datalist],
        commonscale=True,direction=direction))
    fig=plt.gcf()
    multi = MultiCursor(fig.canvas, axes, color='r', lw=.5, horizOn=True, vertOn=True)
    plt.title('6 legendre removed')
    
    if outfile is not None:
        if os.path.exists(fn_add_subfix(outfile,'_markers','.dat')):
            np.genfromtxt(fn_add_subfix(outfile,'_markers','.dat'))
        
    for a in axes:
        a.images[0].set_clim(-0.01,0.01)
        add_clickable_markers2(ax=a,propertyname=propertyname,block=True)

    #display(plt.gcf())

    #Open interface. if there is a markers file, read it and plot markers.
    #when finished store markers

    for a in axes:
        logger.info('axis: '+str(a))
        logger.info('markers: '+str(a.markers))
        print(a.markers)
        
    if outfile is not None:
        np.savetxt(fn_add_subfix(outfile,'_markers','.dat'),np.array([a.markers for a in axes]))
        plt.savefig(fn_add_subfix(outfile,'_markers','.png'))
        
    return axes
    

'''
def compare_2images(data,ldata,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False):
    """plot two images from data and ldata on a two panel figure with shared zoom.
    x and y must be the same for the two data sets.
    """

    d1mean,d1std=np.nanmean(data),np.nanstd(data)
    d2mean,d2std=np.nanmean(ldata),np.nanstd(ldata)    
    std=min([d1std,d2std])
    
    if x is None:
        x=np.arange(data.shape[1])
    if y is None:
        y=np.arange(data.shape[0])
        
    plt.figure(fignum)
    
    plt.clf()
    ax1=plt.subplot(121)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    if titles is not None:
        plt.title(titles[0])    
    s=(std if commonscale else d1std)
    axim=plt.imshow(data,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect='auto',vmin=d1mean-s,vmax=d1mean+s)
    plt.colorbar()

    ax2=plt.subplot(122,sharex=ax1,sharey=ax1)
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    if titles is not None:
        plt.title(titles[1])
    s=(std if commonscale else d2std)
    axim=plt.imshow(ldata,extent=np.hstack([span(x),span(y)]),interpolation='None',
        aspect='auto',vmin=d2mean-s,vmax=d2mean+s)
    plt.colorbar()

    return ax1,ax2
'''


def compare_2images(data,ldata,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False):
    """for backward compatibility, replaced by multiplot.compare_images. plot two images from data and ldata on a two panel figure with shared zoom.
    x and y must be the same for the two data sets.
    """
    from plotting.multiplots import compare_images
    return list(compare_images([(data,x,y),(ldata,x,y)],
        aspect='auto',commonscale=True))

def plot_slope_slice(wdata,x,y,scale=(1.,1.,1.),vrange=None,srange=None,filter=False):
    """
    use calculate_slope_2D to calculate slope and
    plot map and respective slope maps in x and y. Return the three axis.
    Set scale to (1,1,1000.) for z in micron, x,y in mm.
    srange is used as slope range for plots. If filter is set,
    data outside of the range are also excluded from rms calculation.
    """
    
    #Note very important that filtering is made pointwise on the 2D slope figure:
    #  all values out of range are masked and all lines containing at least one value are 
    #  excluded.
    #  this is very different than masking on the base of line slice slope rms,
    #  that is the quantity plotted in the histogram.
    
    slopeax,slopeaz=calculate_slope_2D(wdata,x,y,scale=scale)
    if filter:
        slopeax=np.where(np.logical_and(slopeax>srange[0],slopeax<srange[1]),slopeax,np.nan)
        slopeaz=np.where(np.logical_and(slopeaz>srange[0],slopeaz<srange[1]),slopeaz,np.nan)
        psrange=None  #plot scale
    else:
        if srange is not None:
            psrange=[0,max(np.abs(srange))]
        else:
            psrange=None
        
    plt.clf()
    #plt.suptitle('PCO1S16')
    ax1=plt.subplot(221)
    #import pdb
    #pdb.set_trace()
    plt.imshow(wdata,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='auto',origin='lower')
    plt.clim(vrange)
    plt.xlabel('X(mm)')
    plt.ylabel('Y(mm)')
    plt.colorbar()
    ax2=plt.subplot(222,sharey=ax1)
    plt.plot(np.nanstd(slopeaz,axis=1),y)
    isbad=np.isnan(np.nanstd(slopeaz,axis=1)) #boolean array True on cols with nans
    nbad=len(np.where(isbad)[0])
    if nbad > 0:
        plt.plot(np.repeat(ax2.get_xlim()[1],nbad),y[isbad],'>')
    if psrange is not None: plt.xlim(psrange)
    plt.title('total slope rms %6.3f arcsec'%np.nanstd(slopeaz))
    plt.ylabel('Y(mm)')
    plt.xlabel('Slice X Slope rms (arcsec)')
    ax3=plt.subplot(223,sharex=ax1)
    plt.plot(x,np.nanstd(slopeax,axis=0))
    isbad=np.isnan(np.nanstd(slopeax,axis=0)) #boolean array True on cols with nans
    nbad=len(np.where(isbad)[0])
    if nbad > 0:
        plt.plot(x[isbad],np.repeat(ax3.get_ylim()[1],nbad),'^') #plot symbol on invalid points
    if psrange is not None: plt.ylim(psrange)
    plt.title('total slope rms %6.3f arcsec'%np.nanstd(slopeax))
    plt.xlabel('X(mm)')
    plt.ylabel('Slice Y Slope rms (arcsec)')
    plt.tight_layout()
    plt.colorbar().remove() #dirty trick to adjust size to other panels
    #display(plt.gcf())
    ax4=plt.subplot(224)
    #histogram
    plt.hist(np.nanstd(slopeax,axis=0)[~np.isnan(np.nanstd(slopeax,axis=0))],bins=100,label='AX slope',alpha=0.2,color='r',normed=True);
    plt.hist(np.nanstd(slopeaz,axis=1)[~np.isnan(np.nanstd(slopeaz,axis=1))],bins=100,label='AZ slope',alpha=0.2,color='b',normed=True);
    plt.xlabel('Slope rms (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    
    return ax1,ax2,ax3,ax4
    
def plot_slope_2D(wdata,x,y,scale=(1.,1.,1.),vrange=None,srange=None,filter=False):
    """
    use calculate_slope_2D to calculate slope and
    plot map and slice slope rms in x and y. Return the three axis.
    Set scale to (1,1,1000.) for z in micron, x,y in mm.
    If filter is set, data out of srange are removed and automatic scale is
      used for plot. If not, srange is used for plot axis, but all data
      are used in plots and rms calculation. """
    
    slopeax,slopeaz=calculate_slope_2D(wdata,x,y,scale=scale)
    if filter:
        slopeax=np.where(np.logical_and(slopeax>srange[0],slopeax<srange[1]),slopeax,np.nan)
        slopeaz=np.where(np.logical_and(slopeaz>srange[0],slopeaz<srange[1]),slopeaz,np.nan)
        psrange=None  #plot scale
    else:
        psrange=srange
    plt.clf()
    
    #plt.suptitle('PCO1S16')
    ax1=plt.subplot(221)
    plt.imshow(wdata,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='auto',origin='lower')
    plt.xlabel('X(mm)')
    plt.ylabel('Y(mm)')
    plt.colorbar()
    plt.clim(vrange)
    ax2=plt.subplot(222,sharey=ax1)
    plt.imshow(slopeaz,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='auto',origin='lower')
    plt.colorbar().set_label('arcsec', rotation=270)
    plt.clim(psrange)
    plt.title('Azimuthal slope')
    ax3=plt.subplot(223,sharex=ax1)
    plt.imshow(slopeax,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='auto',origin='lower')
    plt.colorbar().set_label('arcsec', rotation=270)
    plt.clim(psrange)
    plt.title('Axial slope')
    ax4=plt.subplot(224)
    #histogram
    plt.hist(slopeax[~np.isnan(slopeax)],bins=100,label='AX slope',alpha=0.2,color='r',normed=True);
    plt.hist(slopeaz[~np.isnan(slopeaz)],bins=100,label='AZ slope',alpha=0.2,color='b',normed=True);
    plt.xlabel('Slope (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    plt.tight_layout()
    return ax1,ax2,ax3,ax4
    
## CASE-SPECIFIC ANALYSIS FUNCTIONS

def levelpoints(w0):    
    """plot and return matrices of wdata, lwdata, lwdata2
    from points w0"""
    
    xwg,ywg=points_find_grid(w0,'grid')[1]
    wdata=resample_grid(w0,xwg,ywg,matrix=1)
    
    return leveldata(wdata,xwg,ywg)
    
## BACKCOMPATIBILITY
def plot_surface_analysis(*args,**kwargs):
    from scripts.plot_surface_analysis import plot_surface_analysis
    print("function plot_surface_analysis was moved to scripts.plot_surface_analysis, please update import in code as from scripts.plot_surface_analysis import plot_surface_analysis.")
    return plot_surface_analysis (*args,**kwargs)
    
def leveldata(*args,**kwargs):
    from scripts.leveldata import leveldata
    print("function leveldata was moved to scripts.plot_surface_analysis, please update import in code as from scripts.plot_surface_analysis import leveldata.")
    return leveldata (*args,**kwargs)    
    
##TEST FUNCTIONS

def test_remove_nan_frame():
    a=np.arange(12).reshape(3,4).astype(float)
    a[0,:]=np.nan
    x=np.arange(4)
    y=np.arange(3)
    from pySurf.data2D import remove_nan_frame
    print('initial:\n',a,x,y)
    
    print("result:\n",remove_nan_frame(a,x,y))
    print("final values (array is nont modified):\n",a,x,y)
    print('Returns values.')
    return a,x,y


    
def test_fails_leveling():
    """reproduce warning about fit"""
    from pySurf.instrumentReader import matrixZygo_reader
    f='G:\\My Drive\\Shared by Vincenzo\\Slumping\\Coating Metrology\\newview\\20180330_slumped\\07_PCO1.3S04.asc'
    wdata,x,y=matrixZygo_reader(f,scale=(1000.,1000,1.),center=(0,0)) 
    #wdata,x,y=a[0]-legendre2d(a[0],2,1)[0],a[1],a[2]  #simple way to remove plane
    rem4=removelegendre(y,4)  #remove first 4 legendre components
    ldata=level_by_line(wdata,rem4)
    
    
    
if __name__=="__main__":
    from pySurf.outliers2d import remove_outliers2d
    #make a test array
    d=np.random.random(20).reshape((5,4))
    d[[3,2],[1,0]]=25
    compare_2images(d,remove_outliers2d(d))
    
    #test avg routines
    p=((np.random.rand(20)*25).astype(int)).reshape(4,5)
    