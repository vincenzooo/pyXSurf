"""Similar to points, but manage matrix data with x and y. All functions accept arguments in form data,x,y where x and y can be optional in some routines.

2016/10/09 copied from PSDanalysis.py, quite rough, it contains also psd functions that should go in psd2d.
IDL differentiate between functions and procedure. Code syntax is rough (do begin end, -> for methods, comparison operators), but very clear to read.
For example, allows to understand if modify object (procedure) or
return a result (function). Nobody forbids to a function to modify the argument (e.g. if I modify an object or a variable inside a function, are these changes reflected outside?), however it is usually (always?) not needed, because there is a procedure for that. This also enforces the user to think to the correct interface when the subroutine is called.
A flag is called as

In python I need to look at code or docstring to understand if a value is modified and this can also not be consistent (some method can work as functions, others as procedures.
self.data=newdata makes a method a procedure, self.copy().data=newdata; return res is a function
"""

import matplotlib.pyplot as plt
import numpy as np
#from pySurf.points import *#from pySurf.psd2d import *
from pyProfile.profile import polyfit_profile
#from plotting.multiplots import compare_images
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span, span_from_pixels
from dataIO.outliers import remove_outliers, EmptyRangeWarning
import os
from scipy.ndimage import map_coordinates
from scipy import interpolate as ip
from scipy import ndimage
from plotting.captions import legendbox
from astropy.io import fits
from pySurf.points import points_find_grid
from pySurf.points import resample_grid, rebin_points

from pySurf.points import points_in_poly, points_autoresample
from plotting.add_clickable_markers import add_clickable_markers2
from pySurf.find_internal_rectangle import find_internal_rectangle
import itertools

from dataIO.functions import update_docstring
from dataIO.arrays import stats, is_nested_list

from pyProfile.profile import PSF_spizzichino,line
from dataIO.functions import update_docstring

rad_to_sec = 180/np.pi*3600.

test_folder = r'C:\Users\kovor\Documents\python\pyXTel\pySurf\test'

class EmptyPlotRangeWarning(EmptyRangeWarning):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

## SEMI-OBSOLETE LEVELING WITH 1D AUXILIARY FUNCTIONS
## STILL ONLY WAY TO HANDLE GENERIC (non-legendre) leveling by line,
##   code needs review for the part where the leveling function needs x,
##   see example
"""This is the old leveling. level_by_line """

def removelegendre(x,deg):
    """Remove degree polyomial, a possible leveling function for leveldata.
    
    Note: this is superseded by levellegendre
    """
    print ("removelegendre is obsolete, use levellegendre instead.")
    lfit=np.polynomial.legendre.legfit
    lval=np.polynomial.legendre.legval
    def rem(y):
        return y-lval(x,lfit(x,y,deg),tensor=0)
    return rem

def removesag(y):
    """Convenience function to remove second degree polyomial from line.
        
    A possible leveling functiondata for leveldata.
    """
    return y-polyfit_profile (y,degree=2)

def removept(y):
    """Convenienve function to remove piston and tilt from line.
        
    A possible leveling functiondata for leveldata.
    """
    #never tested
    y=y-line (y)
    return y-np.nanmean(y)

def level_by_line(data,function=None,axis=0,**kwargs):
    """Remove line through extremes line by line (along vertical lines).
    
    The returned array has 0 at the ends, but not necessarily zero mean.
    If fignum is set, plot comparison in corresponding figure.
    Function is a function of profile vector y that returns a corrected profile.

    Completely useless, can be replaced by np.apply_along_axis or level_points.
    """
    print ("level_by_line is completely useless function, use np.apply_along_axis or level_points.")

    def remove_line(y): #default line removal function
        return y-line(y)

    if function is None:
        function=remove_line

    ldata=np.apply_along_axis( function, axis=axis, arr=data, **kwargs )

    return ldata

#more updated leveling functions

def fitlegendre(x,y=None,deg=None,nanstrict=False,fixnans=False):
    """Return a legendre fit of degree deg.
       
    Work with 1 or 2D y (if 2D, each column is independently fit and x is 
    the coordinate of first axis).
    if nanstrict is True, every column containing nan (or Inf) is considered 
    invalid and a column of nan is returned, if False, nan are excluded and fit 
    is calculated on valid points only (note that since columns
    are slices along first index, the option has no effect on 1D data 
    (nans are always returned as nans).
    
    2020/09/16 modified 1D/2D mechanism, where all the code,
    including the part in common 1D/2D was moved to the 2D specific part, 
    while the 1D part is completely delegated to functions in ``pyProfile.profile``.
    """
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

    if len(y.shape)==1: 
        result = polyfit_profile(x,y,degree=deg)
    elif len(y.shape)==2:        
        ## this works for 1D or 2D and fits
        lfit=np.polynomial.legendre.legfit
        lval=np.polynomial.legendre.legval

        goodind=np.where(np.sum(~np.isfinite(y),axis=0)==0)[0] #indices of columns non containing any nan
        ## goodind definition can be adjusted for > 2D , datarec definition is already good thanks to [...,] slice

        #fit legendre of degree `deg`. `lval` transpose data (!) so it must be transposed back if 2D
        result=y*np.nan
        if len(goodind)>0:
            #this applies only to 2D data
            datarec=lval(x,lfit(x,y[...,goodind],deg))
            if len(y.shape)==2:
                datarec=datarec.T
            result[...,goodind]= datarec
        ##
        
    #elif len(y.shape)==2:  # moved out 2020/09/16 
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
    """
    # removed 2020/09/16 to rely on pyProfile
    
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
    """
    
    return result


def levellegendre(x,y,deg,nanstrict=False):
    """Remove degree polyomial by line.
    
    Evolution of leveldata using legendre functions
    that work also 2D. nr. of terms fitted in legendre is deg+1 (piston->deg=0).
    For 2D data, data are passed as second argument (y) and y coordinates passed 
    as first (x) (legendre are leveled along columns). 
    Nanstrict excludes the lines containing nans, 
    otherwise only good points are considered.
    """
    datarec=fitlegendre(x,y,deg,nanstrict=nanstrict)

    #result[...,goodind]= y[...,goodind]-datarec #remove poly by line
    result=y-datarec
    return result #y-datarec


def level_data(data,x=None,y=None,degree=1,axis=None,byline=False,fit=False,*args,**kwargs):
    """
    Level a 2D data array using Legendre polynomials.
    
    Use Ryan Allured's routines to remove degree 2D legendres or levellegendre if leveling by line.
    
    Degree can be scalar (it is duplicated) or 2-dim vector. must be scalar if leveling by line. Note the important difference between e.g. ``degree = 2`` and
      ``degree = (2,2)``. The first one uses degree as total degree, it expands then to xl,yl = [0,1,0,1,2,0],[0,0,1,1,0,2]. The second

    leveling by line (controlled by axis keyword) also handle nans.
    x and y are not used, but maintained for interface consistency.
    fit=True returns fit component instead of residuals
    
    Parameters:
    data (ndarray): The 2D data array to be leveled.
    x (ndarray, optional): X-coordinates, unused but maintained for interface consistency.
    y (ndarray, optional): Y-coordinates, unused but maintained for interface consistency.
    degree (int or tuple): The degree of the Legendre polynomial. Scalar or 2D vector.
    axis (int, optional): Axis along which to level. None for 2D leveling, 0 for vertical, 1 for horizontal.
    fit (bool): If True, returns the fit component; otherwise, returns residuals.

    Returns:
    tuple: Leveled data array, x-coordinates, y-coordinates.

    Raises:
    ValueError: If input parameters are invalid.
    
    TODO specify details of routines and document formulas for better repeatabilities.
    """
    
    from utilities.imaging.fitting import legendre2d

    def _validate_degree(degree, axis):
        if axis is not None and np.size(degree) != 1:
            raise ValueError("Degree must be scalar when leveling by line (axis != None): received {}".format(degree))
        return degree

    def _level_by_line(data, y, degree, *args, **kwargs):
        leg = fitlegendre(y, data, degree, *args, **kwargs)
        return leg

    def _level_plane(data, degree, *args, **kwargs):
        xo, yo = degree if np.size(degree) == 2 else (degree, degree)
        xl, yl = [f.flatten() for f in np.meshgrid(np.arange(xo + 2), np.arange(yo + 1))]
        leg = legendre2d(data, x, y, xl=xl, yl=yl, *args, **kwargs)[0]
        return leg

    if x is None:
        x = np.arange(data.shape[1])
    if y is None:
        y = np.arange(data.shape[0])

    degree = _validate_degree(degree, axis)

    if axis == 0:
        leg = _level_by_line(data, y, degree, *args, **kwargs)
    elif axis == 1:
        leg = _level_by_line(data.T, x, degree, *args, **kwargs).T
    elif axis is None:
        leg = _level_plane(data, degree, *args, **kwargs)
    else:
        raise ValueError("Invalid axis value: {}. Axis must be None, 0, or 1.".format(axis))

    return (leg if fit else data - leg), x, y #fails with byline 

## 2D FUNCTIONS

def level_on_points(data,x,y, points):
    """level for best fit plane passing by points.
    points can be provided as x,y,z or as x,y alone, in that case data are interpolated.
    
    ex: xy = [[-0.025,-0.025], [0.025, -0.025], [0,0.042]]
        data,x,y = level_on_points(data,x,y,xy)
    """
    
    from pySurf.points import _get_plane, extract_profile, level_points, matrix_to_points2, points_autoresample

    if np.array(points).shape[1] == 2:
        psurf = matrix_to_points2(data,x,y) 
        # xy points
        p = []
        for pp in points:
            p.append([extract_profile(psurf,pp)[0]])  #with single point, extract point
        points = np.hstack([np.array(points),np.array(p)])

    pars = _get_plane(points,returnPars=True)[1]
    return points_autoresample(level_points(matrix_to_points2(data,x,y),pars=pars))
    

def transpose_data(data,x,y):
    """Transpose (in matrix sense) data and coordinates, switching x and y, return new data,x,y.

    return a view, see np.ndarray.T and np.ndarray.transpose for details.
    """
    data=data.T
    x,y=y,x
    return data,x,y

def apply_transform(data,x,y,trans=None):
    """Apply a 3D transformation (from Nx3 to Nx3) to data.
    
    TODO: add option to set final resampling grid keeping initial sampling, 
    initial number of points or on custom grid (use points.resample_grid, resample_data).
    """
    from pySurf.points import matrix_to_points2,points_autoresample

    if trans is not None:
        p2=trans(matrix_to_points2(data,x,y))
        data,x,y=points_autoresample(p2)

    return data,x,y

def rotate_data(data,x=None,y=None,ang=0,k=None,center=None,
    fill_value=np.nan,usepoints=False,*args,**kwargs):
    """Rotate anticlockwise by an angle in degree.
    
    Non optimized version using points functions.
    2018/10/31 added k parameters allowing 90 deg rotations with np.rot90. 
    k is the number of anticlockwise rotations about center. 
    Note there is not resampling, so it can be inaccurate if center is not 
    on pixel center.

    rotate_Data was intended to work using scipy.ndimage.interpolation.rotate but this failed.
    Added 2018/12/12
    args and kwargs are passed to the function that handles the rotation.

    See also comments on resampling in ``data2D.apply_transform``.

    rot90 determination of rotated axis can probably be extended to general case, but in the meanwhile
    the implementation based on points offers an accurate interpolation (even if slower),
    can be enabled setting flag ``usepoints``.
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
        from pySurf.points import matrix_to_points2, rotate_points
        p=matrix_to_points2(data,x,y)
        p=rotate_points(p,ang/180*np.pi,center=center,*args,**kwargs)
        #res=resample_grid(p,x,y,matrix=True),x,y  #this is wrong, needs to rescale on grid on rotated range
        #similar to data rotation, for now use autoresample
        res=points_autoresample(p)
        #raise NotImplementedError
        return res

    if k is not None:
        assert ang == 0

        ##alg 1: basic rotation with maximum inplace operations, messes up sign of axis, raises ambiguity on how to plot with plot_data
        #data=np.rot90(data,k,axes=(1,0))
        #return data,-y,x        #note that after this: y is same as original x, while y is a whole new variable.
        #                        #data is instead a view (change of elements affects also original data).

        #alg 3: give up inplace operations for axis, center is implemented and determine axis,
        # data are copied on return, without it (alg 2), rotated data would be still a view of the original data.
        data=np.rot90(data,k,axes=(1,0),*args,**kwargs)
        for i in range(k):
            # xlim,ylim=span(x),span(y)
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
    """Save data as matrix on a file.
    
    
    
    Can save as fits if the extension is .fits,
    but this should be probably moved elsewhere,
    otherwise uses np.savetxt to save as text.
    kwargs are passed to np.savetxt or hdu.writeto
    """
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

def register_data(data,x,y,scale=(1,1,1.),
    strip=False,crop=None,center=None,*args,**kwargs):
    """Get data,x,y and register them using usual set of parameters.

    registering operation are performed in the following order and are:
        scale: scale the three axis of the ``scale`` factor, if sign is changed, reorder.
        strip: if True, strip all nans at the border of the data.
        crop: list of ranges (3 axis) to pass to data2D.crop_data
        center: final position of data center (0,0) in x and y data coordinates, 
        if 2 element, center data coordinates, if 3 elements, center also data.
        This means e.g. that data are first cropped (or cleaned of invalid data) than centered. This means that the value puts in the provided coordinate(s) the center of points after cropping operations.
        unexpected parameters passed to register_data are ignored (*args and **kwargs are not used, just suppress error).

    Note that read_data already calls register (after stripping common arguments) careful not to call twice.
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


def read_data(file,rreader,**kwargs):
    """Read data from a file using a given raw reader ``rreader``, with custom options in ``args, kwargs``.

    The function calls raw reader, but, before this, strips all options that are recognized by register_data, all remaining unkown parameters are passed to rreader.
    Then register_data is called with the previously stored settings (or defaults if not present).

    This was made to hide messy code beyond interface. See old notes below, internal behavior can be better fixed e.g. by using dataIO.dicts.pop_kw and inspect.signature and fixing header interface.

    Old notes say:

        Division of parameters is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers,
        while this can be done using read_data.

        this is an ugly way to deal with the fact that return
        arguments are different if header is set, so when assigned to a variable as in patch routines in pySurf instrumentReader it fails.
        Workaround has been calling directly read_data, not optimal.

        2019/04/09 merged from data2D and instrumentReader to data2D. Mostly code from data2D and comments
            from instrumentReader. code commented with ## was excluded.
            
        non essendo sicuro dell'interfaccia per ora faccio cosi'.
        The function first calls the (raw) data reader, then applies the register_data function to address changes of scale etc,
        arguments are filtered and passed each one to the proper routine.
        18/06/18 add all optional parameters, if reader is not passed,
        only registering is done. note that already if no register_data
        arguments are passed, registration is skipped.
        18/06/18 add action argument. Can be  'read', 'register' or 'all' (default, =read and register). This is useful to give fine control, for example to modify x and y after reading and still make it possible to register data (e.g. this is done in
        Data2D.__init__).    
    """
    ##if kwargs.pop('header',False):
    ##    try:
    ##        h = reader(file,header=True)
    ##        return h if h else ""
    ##    except TypeError:  #unexpected keyword if header is not implemented
    ##        return None

    # return header if available, maybe temporary.
    head=kwargs.get('header',False)
    if head:
        return rreader(file,**kwargs)
    
    #filters register_data parameters cleaning args
    # done manually, can use `dataIO.dicts.pop_kw`.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    #zscale=kwargs.pop('zscale',None) this is removed and used only for reader
    # functions where a conversion is needed (e.g. wavelength)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    ##regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}

    #kwargs=pop_kw(kwargs,['scale','crop','center','strip'],
    #    [(1,1,1),None,None,False])

    #get_data using format_reader
    #pdb.set_trace()
    data,x,y=rreader(file,**kwargs)

    return register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip,**kwargs)

    ## #try to modify kwargs
    ## for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    ## #kwargs.update(regdic)
    ## for k in regdic.keys():kwargs[k]=regdic[k]
    ## if register:
    ##    data,x,y=register_data(data,x,y,scale=scale,crop=crop,
    ##    center=center,strip=strip)
    ## return data,x,y

''' merged 2019/04/09
def read_data(file,reader,register=True,*args,**kwargs):
    #from instrumentReader
    """non essendo sicuro dell'interfaccia per ora faccio cosi'.
    The function first calls the (raw) data reader, then applies the register_data function to address changes of scale etc,
    arguments are filtered and passed each one to the proper routine.
    18/06/18 add all optional parameters, if reader is not passed,
    only registering is done. note that already if no register_data
    arguments are passed, registration is skipped.
    18/06/18 add action argument. Can be  'read', 'register' or 'all' (default, =read and register). This is useful to give fine control, for example to modify x and y after reading and still make it possible to register data (e.g. this is done in
    Data2D.__init__).

    implementation:
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


    #filters register_data parameters cleaning args
    # done manually, there is a function in dataIO.
    scale=kwargs.pop('scale',(1,1,1))
    crop=kwargs.pop('crop',None)
    center=kwargs.pop('center',None)
    strip=kwargs.pop('strip',False)
    regdic={'scale':scale,'crop':crop,'center':center,'strip':strip}

    data,x,y=reader(file,*args,**kwargs)

    #try to modify kwargs
    for k in list(kwargs.keys()): kwargs.pop(k)  #make it empty
    #kwargs.update(regdic)
    for k in regdic.keys():kwargs[k]=regdic[k]
    if register:
        data,x,y=register_data(data,x,y,scale=scale,crop=crop,
        center=center,strip=strip)
    return data,x,y



def read_data(file,reader,*args,**kwargs):

    """read data from a file using a given reader with custom options in args, kwargs.

    The function calls reader, but, before this, strips all options that are recognized by register_data,
      all remaining unkown parameters are passed to reader.
    Then register_data is called with the previously stored settings (or defaults if not present).

    This was made to hide messy code beyond interface. See old notes below, internal behavior can be better fixed e.g. by using dataIO.dicts.pop_kw and inspect.signature and fixing header interface.

    Old notes say:

        Division of parameters is hard coded, that is neither too elegant or maintainable. Note however that with this structure it is
        possible to call the read_data procedure with specific parameters, for example in example below, the reader for
        Zygo cannot be called directly with intensity keyword set to True without making a specific case from the other readers,
        while this can be done using read_data.

        this is an ugly way to deal with the fact that return
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
'''

# this is old version before read_data. Note that x and y have no effect, has a autocrop parameter that removes nans.
# pass options to np.genfromtxt
def get_data(*args,**kwargs):
    """Old version.
    """
    '''
    def get_data(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,
            addaxis=False,center=None,skip_header=None,delimiter=' ',
            autocrop=False):
    '''
    from pySurf.data2D import data_from_txt
    
    """replaced by data_from_txt."""
    print ('this routine was replaced by `data_from_txt`, update code')
    return data_from_txt(*args,**kwargs)

def data_from_txt(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,
    addaxis=False,center=None,skip_header=None,delimiter=' ',strip=False,**kwargs):
    """Read matrix from text file. Return data,x,y.
    
    handle addaxis, center and strip nan, on top of all ``np.genfromtxt`` options.
    This function shouldn't be called directly, there are smarter ways of doing it using read_data and readers,
    however, this is a quick way to get data from text if you don't know what I am talking about.

    center: is the position of the center of the image in final coordinates (changed on 2016/08/10, it was '(before any scaling or rotation) in absolute coordinates.') If None coordinates are left unchanged.
        Set to (0,0) to center the coordinate system to the data.
    addaxis: (if matrix is set) can be set to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_data.
    strip (renamed from autocrop): remove frame of nans (external rows and columns made all of nans),
        note it is done before centering. To include all data in centering, crop data
        at a second time with remove_nan_frame
        
    2020/07/10 Added kwargs even if they are not used to suppress error if unknown arguments
        are passed, that can be convenient when kwargs for multiple subfunctions are passed
        to the calling code. This way, kwargs that are meant for another function are tollerated,
        even if it is suboptimal, a better filtering of kwargs should be done.
        For example, makes it fail when called by instrument_reader.read_data with header=True,
        because even if the function doesn't expect a header keyword, the caller routine doesn't
        detect the error. Solution is to make the calling routine check for undefined values?
    """
    #pdb.set_trace()
    #2014/04/29 added x and y as preferred arguments to xrange and yrange (to be removed).
    if skip_header is None:
        skip=0
    else:
        skip=skip_header
    #pdb.set_trace()
    mdata=np.genfromtxt(filename,skip_header=skip,delimiter=delimiter)
    if addaxis: 
        y,mdata=np.hsplit(mdata,[1])
        y=y[1:].flatten() #discard first corner value
        x,mdata=np.vsplit(mdata,[1])
        x=x.flatten() #remove first dimension
    else:
        x=np.arange(mdata.shape[1])
        y=np.arange(mdata.shape[0])

    if strip:
        mdata,x,y=remove_nan_frame(mdata,x,y)

    if center is not None:
        assert len(center)==2
        x=x-(((np.nanmax(x)+np.nanmin(x))/2)-center[0])
        y=y-(((np.nanmax(y)+np.nanmin(y))/2)-center[0])

    return mdata,x,y


def resample_data(d1,d2,method='mc',onfirst=False):
    """Resample d1 [Ny' x Nx'] on x and y from d2[Nx x Ny].
    
    d1 and d2 are passed as list of data,x,y. 
    Return a [Nx x Ny] data, only x and y are used from d2, pass (None, x, y) if you want to resample on a grid.
    onfirst allow to resample second array on first (same as swapping args).
    To get a (plottable) matrix of data use:
    plt.imshow(rpoints[:,2].reshape(ygrid.size,xgrid.size)).
    """
    
    from scipy.interpolate import interp1d

    if onfirst:
        if d2 is None: raise ValueError("d2 must be set if you use ``onfirst`` flag.")
        d1,d2=d2,d1

    try:
        data1,x1,y1=d1
    except ValueError:
        data1=d1
        x1,y1=np.arange(d1.shape[1]),np.arange(d1.shape[0])

    try:
        _,x2,y2=d2 #grid for resampling. data is useless.
    except ValueError:
        #x and y not defined, use data1.
        #x2,y2=np.linspace(*span(x1),d2.shape[1]),np.linspace(*span(y1),d2.shape[0])
        # see https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable#:~:text=Checking%20isinstance(obj%2C%20Iterable),to%20call%20iter(obj)%20.
        if not hasattr(d2, '__iter__'): #(not isiterable(d2)):
            raise ValueError("d2 must be either Data2D or 2-el list.")
        x2,y2 = d2
        
    # replace following block because of DeprecationWarning: elementwise comparison failed; this will raise an error in the future.
    if np.array_equal(x1,x2) and np.array_equal(y1,y2):  
        return data1,x1,y1  
    # try:
    #     #if data_equal((data1,x1,y1),(data2,x2,y2)):
    #     # x and y should not contain nan, but just in case do full comparison to avoid
    #     # nan != nan
    #     #raise ValueError
    #     #breakpoint()
    #     if ((x1 == x2) | (np.isnan(x1) & np.isnan(x2))).all() & ((y1 == y2) | (np.isnan(y1) & np.isnan(y2))).all():
    #         return data1,x1,y1
    # except ValueError:
    #     pass #elements have different shape, they are different and need resampling

    #data2,x2,y2=d2
    if method=='gd':   #use griddata from scipy.interpolate
        z=ip.griddata(np.array(np.meshgrid(x1,y1)).reshape(2,-1).T,data1.flatten(),np.array(np.meshgrid(x2,y2)).reshape(2,-1).T) #there must be an easier way
        z.reshape(-1,len(x2))

        #old not working z=ip.griddata(np.meshgrid(x1,y1),data1,(x2,y2),method='linear') #this is super slow,
    elif method == 'mc':  #map_coordinates
        def indint (x1,x2):
            """Interpolate x2 on x1 returning interpolated index value.
            
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

def rebin_data(d1,d2,method='mc',onfirst=False,*args,**kwargs):
    """Rebin data from d1 to d2. Return a sequence of resampled data `data`, `x`, `y`.
    
    Rebin d1 [Ny' x Nx'] on x and y from d2[Nx x Ny] using `stats.binned_statistic_2d` (additional args can be passed).
    d1 and d2 are passed as list of data,x,y.
    Return a [Nx x Ny] data, only x and y are used from d2, pass (None, x, y) if you want to rebin on a grid.
    
    onfirst allow to resample second array on first (same as swapping args).
    It can give rise to artifacts if you are not careful with the lateral sampling (x and y) because it can have empty bins.
    
    Uses rebin_points from pySurf.points.
    This is a wrapper for rebin_points, that uses stats.binned_statistic_2d.
    
    Example of usage:
    
        d1 = np.random.rand(100, 100)
        d2 = np.random.rand(50, 50)
        data, x, y = rebin_data(d1, d2)
    
    """        
    from pySurf.points import matrix_to_points2
    
    if onfirst:
        if d2 is None: raise ValueError("d2 must be set if you use ``onfirst`` flag.")
        d1,d2=d2,d1
    
    return rebin_points(matrix_to_points2(*d1), matrix=True, bins=[d2[1],d2[2]],*args,**kwargs)


def subtract_data(d1,d2,xysecond=False,resample=True):
    """d1 and d2 are triplets (data,x,y), second array is automatically resampled
    on first, unless resample is set to False (in that case, data arrays are assumed
    having same size and subtracted.
    If xySecond is set to True results are calculated on xy of 2nd array,
    equivalent to -(subtract_data(d2-d1)).
    """
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
    
    If xySecond is set to True results are calculated on xy of 2nd array.
    """
    if resample: d2=resample_data(d2,d1,onfirst=xysecond)

    return d1[0]+d2[0],d1[1],d1[2]

def grid_in_poly(x,y,verts):
    """Data is not needed. Example of usage?"""
    
    p=np.array([xy.flatten() for xy in np.array(np.meshgrid(x,y))]).T
    return np.reshape(points_in_poly(p,verts),(len(y),len(x)))

def crop_data(data,x,y,xrange=None,yrange=None,zrange=None,mask=False,poly=None,
    interactive=False,*args,**kwargs):
    """Return data,x,y of cropped data inside axis ranges, polygons, or interactively
    selected rectangular region.

    axis ranges are passed as a 2-element vector of which each can be None
    or None, where None indicates automatic range (adjust to data).
    If mask is set to True, return a boolean mask of the cropped region.
    poly is a list of vertex for a polygon.

    If interactive is True, allows interactive selection with:
    Zoom to the region to crop, and/or use CTRL+leftClick to add points and create
    an polygonal selection. CTRL+rightClick remove the nearest point. Press ENTER when done.
    """
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
        np.where([xx is None for xx in zrange],span(data),zrange)

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
    """Original version before adding polygonal and interactive selection.
    """
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
        zrange = np.where([xx is None for xx in zrange],span(data),zrange)

    #spdb.set_trace()
    data=data[:,(x>=xrange[0]) & (x<=xrange[1])]
    data=data[(y>=yrange[0]) & (y<=yrange[1]),:]
    x=x[(x>=xrange[0]) & (x<=xrange[1])]
    y=y[(y>=yrange[0]) & (y<=yrange[1])]
    data[((data>zrange[1]) | (data<zrange[0]))]=np.nan
    return data,x,y

def remove_nan_frame(data,x,y,internal=False):
    """Remove all external rows and columns that contains only nans. 
    If internal is set, return the
    internal crop (largest internal rectangle without nans on the frame).
    """
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
    span: if set, return  3 vectors [avg, min, max] with min and max calculated 
    pointwise along same direction.
    expand: instead of a single vector with point-wise minima, returns lists of 
    all vectors having at least one point that is minimum (maximum) between all vectors parallel to axis. Overrides span. ex:
    
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

    print("WARNING: calculate_slope_2D doesn't have a standard interface and will be removed, use slope_2D")

    dx=np.diff(x)#[:,np.newaxis]
    dy=np.diff(y) [:,np.newaxis]  #original

    grad=np.gradient(wdata)  #0->axial(along y), 1->az (along x)
    slopeax=grad[0][:-1,:]/scale[-1]*rad_to_sec/dy  #in arcseconds from um  #original
    slopeaz=grad[1][:,:-1]/scale[-1]*rad_to_sec/dx
    #slopeax=grad[0][:,:-1]/scale[-1]*rad_to_sec/dy  #in arcseconds from um
    #slopeaz=grad[1][:-1,:]/scale[-1]*rad_to_sec/dx

    return slopeax,slopeaz

def slope_2D(wdata,x,y,scale=(1.,1.,1.)):
    """calculate slope maps in x and y.
    return a couple of maps of type slope,x,y data for x and y slopes respectively.
    Set scale to (dx,dy,1000.) for z in micron, x,y in mm. (?does it mean  1,1,1000?)"""

    dx=np.diff(x)#[:,np.newaxis]
    dy=np.diff(y) [:,np.newaxis]  #original

    grad=np.gradient(wdata)  #0->axial(along y), 1->az (along x)
    slopeax=grad[0][:-1,:]/scale[-1]*rad_to_sec/dy  #in arcseconds from um  #original
    yax=y[:-1]+dy.ravel()/2

    slopeaz=grad[1][:,:-1]/scale[-1]*rad_to_sec/dx
    xaz=x[:-1]+dx/2
    #slopeax=grad[0][:,:-1]/scale[-1]*rad_to_sec/dy  #in arcseconds from um
    #slopeaz=grad[1][:-1,:]/scale[-1]*rad_to_sec/dx

    return (slopeax,x,yax),(slopeaz,xaz,y)


def get_stats(data=None,x=None,y=None,units=None,vars=None,string=False,fmt=None):
    """ Return selected statistics for each of data,x,y as numeric array or string, wrapping ``dataIO.stats``. 
    
    ``vars`` determines which statistical indicators are included in stats, while ``string``, ``fmt`` and ``units`` are used to generate and control a string output in a similar way as in wrapped function ``dataIO.arrays.stats``. ``get_stats`` implements a more versatile syntax handling statistics on  three coordinates axis.
    See the test function ``test_get_stats`` for more examples.
    
    ``vars`` is an array of indices selecting which variables must be included in the statistics. You can call ``stats`` with no argument to see the list of variables (and standard format).    
    
    The options are:
    
    * scalar: use a preset (=1 basic statistics, =2 for extended statistics) single level list of integer indices (e.g. =[0,1]): it is applied only to data. 
    * two-level nested list (e.g. =[[0,1]]) and the outer list has a single element, the selection is used replicated to data, x, y. 
    * 3-element nested list =[[0,1],[1],[2]]: indicates different choices for data, x and y.
    
    In this context, special values can be used to indicate different type of defaults (N.B.: vars are in order matching ``data, x, y``), these are internally converted to the proper format:
    None: don't include element (e.g. [1,2] is equivalent to [[1,2],None,None])
    []:  use default (e.g. [[0,2],[],None] uses default for x and doesn't report y)
    [[]]: use full set of variables (e.g. [[0,2],[],[[]]] uses default for x and full stats for y, [[[]]] uses full for data).

    Statistics are returned as array of numerical values, unless ``string`` flag is set. In that case, ``units`` and ``fmt`` are used to control the output format.
        
    ``units`` (scalar in ``dataIO.arrays.stats``) can be passed as 3-element list of strings to individually set the units for each axis. These are appended to every value in the respective axis (a more flexible behavior can be obtained by using ``fmt``). If scalar is used on data axis, if single element string array, use for all axis (i.e. set ``units`` as array to obtain different behavior like units=['','',u] to set only the data axis).
    
    ``fmt`` uses ``dataIO.arrays.stats``, but it is not divided in axis. All axis settings are combined in a single list. If ``string`` is set to True ``get_stats`` returns a flattened array of strings, so an array of equal lenght can be passed, or a scalar, used for all axis and stats. 
    Note that strings are assembled here without accessing to ``dataIO.arrays.stats`` function, whose ``fmt`` argument is not used at all here. 

    ``units`` are used and appended to ``fmt`` if not None or set to empty string.
    The length of the two must match, and are converted to the correct format inside this function.
    Conversion is made in this case in dependance on the format of ``vars``. For example, ``vars = [[1,2,3],None,None]`` requires to convert ['mm','mm','um'] to ['um','um','um']
    
    If default, units are built from vars and from strings obtained from ``dataIO.arrays.stats`` (called without data).
    
    TODO: span doesn't exclude nan data, put flag to tune this option.
    TODO: there is some confusion in creating labels for ``plot_data`` because it can be unclear which one is X, Y, Z. A label should be added externally or in a routine. Also, statistics cannot be sorted (a list is returned, so it is possible to sort the list).
    TODO: make a default extended stats, with  span and pts nr. for x and y and mean, span, rms for z. 
    
    """
    
    """
    If a single scalar value is passed as ``vars``, this is intended as a preset (1,2 for backward compatibility, where 1 is basic data statistics, and 2 more extended, including x and y size). These presets are defined in ``data2D.get_stats``, which can also be called directly to test generation of legend for ``data2D.plot_data``, like e.g.:

        from pySurf.data2D import get_stats 
        get_stats(data,x,y,vars=[1,2,3],units=['mm','mm','mm'],string=True,fmt=None)
        
    """
    # put value of ``units`` in u, converted to standard 3-el string array 
    if units is None or not string:
        u = ["","",""]
    elif isinstance(units,str): #assumes z
        u = [units,units,units]
    elif len(units) == 2: #x,y, no z
        u = [units[0],units[1],""]
    elif len(units) == 1: # z, no x,y
        #breakpoint()
        u = ["","",units[0]]  
    elif len(units) == 3:
        u = units
    else:  # if 3 units are already ok
        raise ValueError ("Unrecognized units.")
    
    u = [uu if uu is not None else "" for uu in u] # to avoid error is [None, None, None] or similar is passed.
    
    # this handles all special cases and converts vars to a three-el list of lists.
    #   with elements (possibly empty) lists of indices.
    if vars is None:
        vars = [None,[],[]]
    
    if not is_nested_list(vars): 
        if not isinstance(vars,list): #cannot be None is scalar
            assert isinstance(vars,int)
            # presets
            if vars==1: 
                vars = [[1,3],[],[]]  # N.B.: it is stddev even if named rms
                if fmt is None:
                    fmt = ['rms: %.3g '+u[2],'PV: %.3g '+u[2]]
    
            elif vars==2: #backwards compatibility
                vars = [[0,1,3],[6],[6]]   # mean,PV, nx, ny
                if fmt is None:
                    fmt = ['mean: %.3g '+u[2],'PV: %.3g '+u[2],'rms: %.3g '+u[2],'nx:%i', 'ny: %i']
        else:
            if vars[0] is None:
                vars = None
            vars=[vars,[],[]]
    else:
        if len(vars)==1:
            vars = [vars[0],vars[0],vars[0]] #careful, these are pointers to same `vars`
        else:
            assert len(vars)==3 
    #return vars    
    # vars here is in shape [[],[],[]], if preset, also format was set if not provided.
    
    # `fmt` must be a single string or list of strings that joined have
    #   the right number of % placeholders for `st` values.
    # u is a list of 3 units for axis.
    #breakpoint()
    # run with string = True if `fmt` is None, using default stats and getting strings 
    # if fmt is set () string is False (string is never used). In this case an array of floats is returned, to be later processed using `fmt`.
    # units are ignored in this case, but they shoud have been incorporated already in fmt.
        #if string is False and fmt is None, strings are returned.
    #breakpoint()
    st = [stats(d,vars=v,string=(fmt is None) and string,units=uu) for d,v,uu in zip([data,x,y],vars,[u[2],u[0],u[1]])]
    st = list(itertools.chain.from_iterable(st)) # flatten list   
    if data is not None and fmt is not None and string:
        st=(("\n".join(fmt))%tuple(st)).split('\n')
        #st=("\n".join(fmt).format(tuple(st)).split('\n')
    
    return st
get_stats = update_docstring(get_stats, stats)

def test_get_stats(*value,help=True):
    
    data,x,y = value if value else load_test_data()
    
    
    # TODO: wrap the following in decorator, to log
    #    and pring placeholder for ERROR writing details
    #    at the end of test.
    
    if help: print (get_stats.__doc__)
    
    try:
        print("load test surface, shape: ",data.shape)
        print("no options:\n",get_stats(data,x,y),"\n-------------\n")
        # [-0.6937998887775654, 0.3722780252351366, 0.7873685374338596,1.8219939575195314,-1.2262238037109376, 0.5957701538085938,76800]
        
        print("no options, string:\n",get_stats(data,x,y,string=True),"\n-------------\n")
        
        print("-- PRESETS --")
        print("vars=1 (preset), no data:\n",get_stats(string=True,vars=1),"\n-------------\n")
        print("vars=1, /string:\n",get_stats(data,x,y,string=True,vars=1),"\n-------------\n")
        print("vars=2, /string:\n",get_stats(data,x,y,string=True,vars=2),"\n-------------\n")
        print("vars=1, string:False\n",get_stats(data,x,y,string=False,vars=1),"\n-------------\n")
        
        print("-- FLAG STRING --")
        print("vars=[1,2,3]:\n",get_stats(data,x,y,string=True,vars=[1,2,3]),"\n-------------\n")
        #['StdDev: 3.1 ', 'PV: 35.2 ', 'min: -32.6 ']
        
        print("vars=[[1,2,3]]:\n",get_stats(data,x,y,string=True,vars=[[1,2,3]]),"\n-------------\n")
        #[['StdDev: 3.1 ', 'PV: 35.2 ', 'min: -32.6 '],
        #['StdDev: 28.2 ', 'PV: 97.4 ', 'min: -48.7 '],
        #['StdDev: 29.4 ', 'PV: 102 ', 'min: -50.8 ']]
        
        #this works
        print("vars=[3]:\n",
              get_stats(data,x,y,string=True,vars=[3]),"\n-------------\n")
        #[['min: -32.6 '], ['min: -48.7 '], ['min: -50.8 ']] 
        #this fails:
        #print(get_stats(data,x,y,string=True,vars=3),"\n-------------\n")

        print("vars=[[],[2],[3]]:\n",
              get_stats(data,x,y,string=True,vars=[[],[2],[3]]),"\n-------------\n")
        #[['StdDev: 3.1 '], ['PV: 97.4 '], ['min: -50.8 ']]
           
        print("vars=[[1],[2],[3]]:\n",
              get_stats(data,x,y,string=True,vars=[[1],[2],[3]]),"\n-------------\n")
        #[['StdDev: 3.1 '], ['PV: 97.4 '], ['min: -50.8 ']]
        
        print('fmt and alls parameters are ignored if string is False')
        print(get_stats(data,x,y,string=False,vars=[[2,6],[],[1]],units='km',fmt=''))
        

        print("-- UNITS --")
        
        print("units=['km'] with default vars:\n",
              get_stats(data,x,y,string=True,units=['km']),"\n-------------\n")
        
        # units with preset
        print("units='mm' with preset 1:\n",
            get_stats(data,x,y,string=True,vars=1,units='mm'),"\n-------------\n")

        print("units='[mm]':\n",
              get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['mm']),"\n-------------\n")
        #[['StdDev: 3.1 mm'], ['PV: 97.4 '], ['min: -50.8 ']]
        
        print("units=['um','cm','mm']:\n",
              get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['um','cm','mm']),"\n-------------\n")
        #[['StdDev: 3.1 mm'], ['PV: 97.4 mm'], ['min: -50.8 mm']]

        print("vars=[[1],[2,3],[3]]:\n",
              get_stats(data,x,y,string=True,vars=[[1],[2,3],[3]],units=['um','cm','mm']),"\n-------------\n")
        #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
        
        print("vars=[[2,3],[],[3]]:\n",
              get_stats(data,x,y,string=True,vars=[[2,3],[],[3]],units=['um','mm','cm']),"\n-------------\n")
        #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
        
        '''
        print("vars=[[2,3],[],[3]]:\n",
              get_stats(data,x,y,string=True,vars=[[2,3],[],[3]],units=['%%%','%%','%']),"\n-------------\n")
        #2022/10/7 fails because % is interpreted in string format, double percent solves, but doesn't work in plot labels.
        '''
    
    
    except:
        raise

'''
def test_get_stats(*value):
    import pprint as pprint
    
    data,x,y = value if value else load_test_data()
    
    print (get_stats.__doc__)
    
    print("load test surface, shape: ",data.shape)
    print("no options:\n",get_stats(data,x,y),"\n-------------\n")
    #print(get_stats(data,x,y,string=True))
    print("no options, string:\n",get_stats(data,x,y,string=True),"\n-------------\n")
    
    print("vars=[1,2,3], string:\n",get_stats(data,x,y,string=True,vars=[1,2,3]),"\n-------------\n")
    #[['StdDev: 3.1 ', 'PV: 35.2 ', 'min: -32.6 '],
    #['StdDev: 28.2 ', 'PV: 97.4 ', 'min: -48.7 '],
    #['StdDev: 29.4 ', 'PV: 102 ', 'min: -50.8 ']]
    
    #this works
    print("vars=[3], string:\n",
          get_stats(data,x,y,string=True,vars=[3]),"\n-------------\n")
    #[['min: -32.6 '], ['min: -48.7 '], ['min: -50.8 ']] 
    #this fails:
    #print(get_stats(data,x,y,string=True,vars=3),"\n-------------\n")

    print("vars=[[],[2],[3]], string:\n",
          get_stats(data,x,y,string=True,vars=[[],[2],[3]]),"\n-------------\n")
    #[['StdDev: 3.1 '], ['PV: 97.4 '], ['min: -50.8 ']]
       
    print("vars=[[1],[2],[3]]:\n",
          get_stats(data,x,y,string=True,vars=[[1],[2],[3]]),"\n-------------\n")
    #[['StdDev: 3.1 '], ['PV: 97.4 '], ['min: -50.8 ']]


    # units have effect only if provided as 3-el string
    print("units='mm':\n",
          get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units='mm'),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 '], ['min: -50.8 ']]

    print("units='[mm]':\n",
          get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 '], ['min: -50.8 ']]
    
    print("units=['mm','mm','mm']:\n",
          get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm'], ['min: -50.8 mm']]

    print("vars=[[1],[2,3],[3]]:\n",
          get_stats(data,x,y,string=True,vars=[[1],[2,3],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
    
    print("vars=[[2,3],[],[3]]:\n",
          get_stats(data,x,y,string=True,vars=[[2,3],[],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
'''

'''
## Differenza tra data_histostats e plot_stats ?? 
## Rimosso 2020/11/05

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

    if np.array(datalist).size == 1: #non funziona mica!
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
    '''

def data_histostats(data,x=None,y=None,bins=100,density=True,units=None,loc=0,*args,**kwargs):
    """wrapper around plt.hist, plot histogram of data (over existing window) adding label with stats. 
    ``density`` set tu True normalizes distribution to have sum equal 1. Return 3-uple according to ``plt.hist``:"""
    
    units=units if units is not None else ['[X]','[Y]','[Z]']
    #pdb.set_trace()
    res = plt.hist(data[np.isfinite(data)],bins=bins,density=density,*args,**kwargs)
    plt.xlabel('Units of '+units[2])
    plt.ylabel('Fraction of points #')
    
    stats = get_stats(data,x,y,units=units,string=True)
    legend=stats #["\n".join(s) for s in stats] 
        
    # get_stats returns a string and it's not very flexible
    # maybe with a better implementation could add styles
    # for plotting of vertical lines on stats.
    plt.axvline(np.nanmean(data),label='mean: %6.3f'%np.nanmean(data),ls = '--')
    l=legendbox(legend,loc=loc,framealpha=0.2)
    
    plt.axvline(np.nanmean(data)-np.nanstd(data),ls=':')
    plt.axvline(np.nanmean(data)+np.nanstd(data),ls=':')
    #plt.legend(loc=0)
    return res
#data_histostats=update_docstring(data_histostats,plt.hist)


## PLOT FUNCTIONS

def plot_data(data,x=None,y=None,title=None,outfile=None,units=None,stats=False,vars=None,loc=0,contour=False,colors=None,largs=None,framealpha=0.5,nsigma=None,axis_check=True,*args,**kwargs):
    """Plot data using imshow and modifying some default properties.
    Units for x,y,z can be passed as 3-el array or scalar, None can be used to ignore unit.
    Broadcast all imshow arguments.
    stats is a flag to print stats on plot. Here it can be assigned booleanTrue
    or integer value: True or 1 plots statistics, 2 plots x and y spans
    largs is a dictionary of arguments passed to caption in plotting.caption.legendbox
    (used only if stats are plotted).
    (e.g. {'color':'r'}
    Returns axis (modified 2023/01/17, was returning axim as returned by plt.imshow).
    nsigma set colorscale to this multiple of data standard deviation.
    In alternative can be a dictionary containing arguments for remove_outliers. 
    If None (default) range is not changed from matplotlib defaults.
    If dict, a nummber of parameters for can be passed to remove_outliers.remove_outliers to determine color range (data are left intact).
    if axis_check is set, x and y axis must be increasing.
    
    2020/11/05 updated all stats functions.
    2020/07/14 added flag ``contour`` to overplot contours, and colors,
    to be passed to ``plt.contour``"""

    
    ## SET PLOT OPTIONS
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

    if nsigma is not None:
        #pdb.set_trace()
        #questo in qualche modo intercettava l'errore in `remove_outliers`
        #  ma non funzionava quando il range erea buono.
        #with warnings.catch_warnings(record=True) as w:  

        if isinstance(nsigma,dict): #if more than one option were passed
            clim=span(np.where(remove_outliers(data,**nsigma),data,np.nan))
        else:
            clim=span(np.where(remove_outliers(data,nsigma=nsigma),data,np.nan))

        if len(clim)==0: 
            print('Range after filtering was empty, plotting full set of data.')
            clim=span(data)
            
        #plt.clim(*clim)
    #print('clim',clim)
    else:
        clim=[None,None]  #set to None to handle with vmin, vmax

    ## SET AXIS COORDINATES
    #plotting is here to resolve conflict with clim
    if x is None: x = np.arange(data.shape[1])
    if y is None: y = np.arange(data.shape[0])
    if axis_check:
        assert x[-1]>=x[0] and y[-1]>=y[0]
    
    #if one of the two is a single value, plot fails. To handle this special case,
    # single values are duplicated.
    #if np.size(x) == 1: x = x[None, :]
    
    delta = 0.1 # used if one of the axis has a single value.
    # this is a workaround of setting a fake duplicated coordinate
    # and adjust ticks accordingly.
    sx=span_from_pixels(x,delta=delta)  #determines the extent of the plot from 
    sy=span_from_pixels(y,delta=delta)  #  coordinates of pixel centers.
    # import pdb
    # pdb.set_trace()
    vmin=kwargs.pop('vmin',clim[0])
    vmax=kwargs.pop('vmax',clim[1])
    
    fmt = kwargs.pop('fmt',None)
    # to use with `stats` (passed as parameter)
    
    ## PLOT
    # this fails if unknown kwargs is passed:
    plt.imshow(data,extent=(sx[0],sx[1],sy[0],sy[1]),
            aspect=aspect,origin=origin,interpolation='none',
            vmin=vmin, vmax=vmax,**kwargs) 
    
    axim = plt.gca()
    # adjust for the specific case of a single value on x or y axis
    #pdb.set_trace()
    if span(x,size=True) == 0:
        plt.gca().set_xticks(x)
    if span(y,size=True) == 0:
        plt.gca().set_yticks(y)
    #axim=plt.imshow(data,extent=(span(x)[0]-dx,span(x)[1]+dx,span(y)[0]-dy,span(y)[1]+dy),
    #    **pop_kw(kwargs,{'aspect':'equal',
    #                'origin':"lower",
    #                'interpolation':"none"}))
    #print(kwargs)

    ## AXIS LABELS
    plt.xlabel('X'+(" ("+units[0]+")" if units[0] else ""))
    plt.ylabel('Y'+(" ("+units[1]+")" if units[1]  else ""))
    
    ## ADJUST COLORBAR:\
    '''
    #im_ratio = data.shape[0]/data.shape[1]   #imperfect solution found on stack overflow. Doesn't work in some cases (3 panels? logaritmic axis?)
    
    ax=plt.gca()  #try my variation
    s=ax.get_position().size

    
    #im_ratio = data.shape[0]/data.shape[1]   #imperfect solution
    im_ratio = s[0]/s[1] # attempt improvement
    ## im_ratio aggiusta cb a occupare frazione di ?
    ## ma puo' dare problemi per aree peculiari    
    cb = plt.colorbar(axim,fraction=0.046*im_ratio, pad=0.04)
    '''
    
    cb = plt.colorbar()
    '''
    divider = make_axes_locatable(plt.gca())
    bax = divider.append_axes("right", size="5%", pad=0.05) 
    cb = plt.colorbar(axim,cax=bax) 
    '''

    if units[2] is not None:
        cb.ax.set_title(units[2])
        #cb.ax.set_xlabel(units[2],rotation=0)
        #cb.ax.xaxis.set_label_position('top')
        #cb.ax.xaxis.label.set_verticalalignment('top') #leave a little bit more room
    """
    
    ### 2021/04/11 new version using divider. Try to precent problem with
    #        additional axis sized on the base of first ones.

    divider = make_axes_locatable(plt.gca())
    ax1     = divider.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(axim,cax=ax1)  # fondamentale qui usare cax
    """
    
    # goes after colorbar
    if contour: plt.contour(x,y,data,colors=colors,**kwargs)
    
    ## LEGEND BOX   
    #print('STATS:\n',stats) 
    
    # if vars is single scalar, use preset
    if stats:          
        s = get_stats(data,x,y,vars=stats,units=units,string=True,fmt=fmt)
        l=legendbox(s,loc=loc,framealpha=framealpha,draggable=True)
    
    if title is not None:
        plt.title(title)
        
    # import pdb
    # pdb.set_trace()
    if outfile is not None: #useless, just call plt.savefig
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

def test_plot_stats(value=None):
    """Test plotting of legendbox in plot_data"""
    
    
    if value is None:
        data,x,y = load_test_data()
    
    #print (get_stats.__doc__)
    plt.figure()
    plot_data(data,x,y, aspect = 'auto',title = 'no options')

    plt.figure()
    plot_data(data,x,y, aspect = 'auto', stats = 1 ,title = 'backwards comp., stats = 1')   
    
    plt.figure()
    plot_data(data,x,y, aspect = 'auto', stats = 2 ,title = 'backwards comp., stats = 2')
    
    plt.figure()
    plot_data(data,x,y, aspect = 'auto',stats=[1,2,3], title = 'stats = [1,2,3]')
    
    plt.figure()
    plot_data(data,x,y, aspect = 'auto',stats=[[1],[2],[3]], title = 'stats = [[1],[2],[3]]')    
    
    plt.figure()
    plot_data(data,x,y, aspect = 'auto',stats=[[1],[0,2],[3]], title = 'stats = [[1],[0,2],[3]]')  
    
    '''

       
    print(get_stats(data,x,y,string=True,vars=[[1],[2],[3]]),"\n-------------\n")
    #[['StdDev: 3.1 '], ['PV: 97.4 '], ['min: -50.8 ']]

    # units have effect only if provided as 3-el string
    print(get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units='mm'),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 '], ['min: -50.8 ']]

    print(get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 '], ['min: -50.8 ']]
    
    print(get_stats(data,x,y,string=True,vars=[[1],[2],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm'], ['min: -50.8 mm']]

    print(get_stats(data,x,y,string=True,vars=[[1],[2,3],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
    
    print(get_stats(data,x,y,string=True,vars=[[],[2,3],[3]],units=['mm','mm','mm']),"\n-------------\n")
    #[['StdDev: 3.1 mm'], ['PV: 97.4 mm', 'min: -48.7 mm'], ['min: -50.8 mm']]
    '''
    
def psf2d(y,wdata,alpha,xout,nskip=1):
    """return a 2d psf for axial profiles on wdata with coordinate y.
    """
    psf2d=[]
    for col in wdata[:,::nskip].T:
        yout=PSF_spizzichino(y,col/1000.,alpha=alpha,xout=xout)[1]
        psf2d.append(yout)
    return np.array(psf2d).T
    #return xout*rad_to_sec.,np.array(yout)

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

    # TODO: plot of y axis slope (ax3) is not aligned with surface plot.
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
    ax1=plt.subplot(221)
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
    #plt.tight_layout()
    #cb = plt.colorbar() #will be removed at the end, dirty trick to adjust size to other panels
    
    #display(plt.gcf())

    ax4=plt.subplot(224)
    #histogram
    plt.hist(np.nanstd(slopeax,axis=0)[~np.isnan(np.nanstd(slopeax,axis=0))],bins=100,label='AX slope',alpha=0.2,color='r',density=True);
    plt.hist(np.nanstd(slopeaz,axis=1)[~np.isnan(np.nanstd(slopeaz,axis=1))],bins=100,label='AZ slope',alpha=0.2,color='b',density=True);
    plt.xlabel('Slope rms (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    plt.tight_layout()
    #plt.colorbar(ax=ax3).remove()

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
    plt.title('Horizontal slope')

    ax3=plt.subplot(223,sharex=ax1)
    plt.imshow(slopeax,extent=(span(x)[0],span(x)[1],span(y)[0],span(y)[1]),aspect='auto',origin='lower')
    plt.colorbar().set_label('arcsec', rotation=270)
    plt.clim(psrange)
    plt.title('Verticall slope')

    ax4=plt.subplot(224)
    #histogram
    plt.hist(slopeax[~np.isnan(slopeax)],bins=100,label='AX slope',alpha=0.2,color='r',density=True);
    plt.hist(slopeaz[~np.isnan(slopeaz)],bins=100,label='AZ slope',alpha=0.2,color='b',density=True);
    plt.xlabel('Slope (arcsec)')
    plt.ylabel('Fraction of points')
    #plt.title ('rms: %5.3f, PV: %5.3f'%(np.nanstd(tmp[:,2]),span(tmp[:2],1)))
    plt.legend(loc=0)
    #display(plt.gcf())
    plt.tight_layout()
    
    #plt.colorbar(ax=ax4).remove()  #questo sistema il 4o plot, ma incasina il 3o.
    # il problema si verifica alla rimozione della cb, infatti rimpiazzando `plt.tight)layout` con:
    # cb = plt.colorbar(ax=ax4)
    # plt.tight_layout()
    # #cb.remove()  # with this commented, alignment is ok. uncommenting it will mess up 3rd plot 
    
    
    return ax1,ax2,ax3,ax4

## CASE-SPECIFIC ANALYSIS FUNCTIONS

def levelpoints(w0):
    """plot and return matrices of wdata from points w0"""

    xwg,ywg=points_find_grid(w0,'grid')[1]
    wdata=resample_grid(w0,xwg,ywg,matrix=1)

    return leveldata(wdata,xwg,ywg)

## BACKCOMPATIBILITY
def plot_surface_analysis(*args,**kwargs):
    from pySurf.scripts.plot_surface_analysis import plot_surface_analysis
    print("function plot_surface_analysis was moved to scripts.plot_surface_analysis, please update import in code as from scripts.plot_surface_analysis import plot_surface_analysis.")
    return plot_surface_analysis (*args,**kwargs)

def leveldata(*args,**kwargs):
    from pySurf.scripts.plot_surface_analysis import leveldata
    print("function leveldata was moved to scripts.plot_surface_analysis, please update import in code as from scripts.plot_surface_analysis import leveldata.")
    return leveldata (*args,**kwargs)

##TEST FUNCTIONS

def load_test_data():
    """load a standard Zygo file for tests."""
    
    from pathlib import PureWindowsPath,Path
    from pySurf.readers.instrumentReader import matrixZygo_reader
    
    relpath=PureWindowsPath(r'test\input_data\zygo_data\171212_PCO2_Zygo_data.asc')
    wfile= Path(os.path.dirname(__file__)) / relpath
    (d1,x1,y1)=matrixZygo_reader(wfile,ytox=220/1000.,center=(0,0))
    
    return d1,x1,y1

def test_leveling(d=None):
    """plot a set of images, representative of data leveled with different combinations of parameters.
    d is Data2D object.
    """
    from IPython.display import display
    #from pySurf.data2D import load_test_data,plot_data
    
    #get_ipython().run_line_magic('matplotlib', 'inline')
    
    if d is None:
        data,x,y = load_test_data()
    else:
        data,x,y = d
        
    #from pySurf.data2D import level_data,plot_data
    plt.close('all')
    plt.figure()
    plot_data(*level_data(data,x,y,5),stats=True,title='level, 5 degree scalar arg.') #,byline=True
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,(5,5)),stats=True,title='level, (5,5) degree scalar arg.') #,byline=True
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,(0,5)),stats=True,title='level, (0,5) degree arg.')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,5,axis=0),stats=True,title='level, 5 degree arg., axis=0')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,5,axis=1),stats=True,title='level, 5 degree arg., axis=1')

    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,5,fit=True),stats=True,title='fit, 5 degree scalar arg.')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,(5,5),fit=True),stats=True,title='fit, (5,5) degree scalar arg.')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,(0,5),fit=True),stats=True,title='fit, (0,5) degree arg.')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,5,axis=0,fit=True),stats=True,title='fit, 5 degree arg., axis=0')
    display(plt.gcf())
    plt.clf()
    plot_data(*level_data(data,x,y,5,axis=1,fit=True),stats=True,title='fit, 5 degree arg., axis=1')
    display(plt.gcf())
    
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

def outliers_analysis(data,x=None,y=None,nsigma=3,itmax=5,outname=None):
    """Perform multiple iterations of outlier removal, plotting data and histogram at each step."""
    from dataIO.outliers import filter_outliers
        
    for i,m in enumerate(filter_outliers(data,nsigma=nsigma,itmax=itmax)):
        print('iter: ',i)
        print('outliers: ',len(np.where(~m)[0]))
        #print('values: ',a[m])
        #print('\n')
        if i > 0:
            for ii,jj in zip(*(~m).nonzero()):
                plt.plot(x[jj],y[ii],'x')
            if outname:
                plt.savefig(fn_add_subfix(outname,'_%04i'%i,'.png'))
        plt.figure()
        plt.subplot(212)
        data_histostats(data)
        # if very distributed, use log scale
        if span(data,size=True) > np.nanstd(data)*20:
            plt.semilogy()
        
        plt.subplot(211)
        data[~m]=np.nan
        plot_data(data,x,y,stats=2,aspect='auto')


def test_outliers_analysis():
    """run outliers analysis and save output of tests."""
    outfolder = os.path.join(test_folder,r'results\outliers_analysis')
    fn = os.path.join(test_folder,r'input_data\csv\residuo MPR rispetto a parabola nominale Bianca.txt')
    from pySurf.readers.instrumentReader import points_reader
    data,x,y = points_reader(fn,delimiter='',skip_header=2)
        
    outliers_analysis(data,x,y,nsigma=3,itmax=10,outname = os.path.join(outfolder,os.path.basename(fn)))


def test_fails_leveling():
    """reproduce warning about fit"""
    from .readers.instrumentReader import matrixZygo_reader
    f = os.path.join(test_folder,r'input_data\newview\newview\20180330_slumped\07_PCO1.3S04.asc')
    wdata,x,y=matrixZygo_reader(f,scale=(1000.,1000,1.),center=(0,0))
    #wdata,x,y=a[0]-legendre2d(a[0],2,1)[0],a[1],a[2]  #simple way to remove plane
    rem4=removelegendre(y,4)  #remove first 4 legendre components
    ldata=level_by_line(wdata,rem4)
    
    
def test_plot_data_aspect():
    """test for plots with different aspect ratios."""
    a = np.arange(3000).reshape(60,50)
    plt.figure('regular, aspect=equal')
    plot_data(a)
    plt.figure('regular, aspect=auto')
    plot_data(a,aspect='auto')    
    
    a = np.arange(3000).reshape(6,500)
    plt.figure('wide, aspect=equal')
    plot_data(a)
    plt.figure('wide, aspect=auto')
    plot_data(a,aspect='auto')
    
    a = np.arange(3000).reshape(500,6)
    plt.figure('tall, aspect=equal')
    plot_data(a)
    plt.figure('tall, aspect=auto')
    plot_data(a,aspect='auto')



if __name__=="__main__":
    from pySurf.outliers2d import remove_outliers2d
    #make a test array
    d=np.random.random(20).reshape((5,4))
    d[[3,2],[1,0]]=25
    compare_2images(d,remove_outliers2d(d))

    #test avg routines
    p=((np.random.rand(20)*25).astype(int)).reshape(4,5)
