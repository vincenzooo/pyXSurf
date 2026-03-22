from __future__ import print_function
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import interpolate as ip
import os
import pdb
from dataIO.span import span,span_from_pixels
from pySurf.plane_fit import plane_fit
from astropy.io import fits
#from matplotlib.mlab import griddata
#from scipy.interpolate import griddata
from dataIO.running_mean import running_mean
from scipy import stats
from plotting.add_clickable_markers import add_clickable_markers2


verbose = False

"""
Module containing functions acting on a point cloud. With the intention
of creating a class. Points are in format (Npoints,Ndim).
Note also that shape, when used (mostly is just a visual parameter for plots), is in format (nx,ny), that is opposite to python convention.
"""

defaultmethod='linear' #used for ip.griddata (scipy.interpolate.griddata)
# note from scipy.interpolate import interp2d to interpolate from grid to grid
# also scipy.ndimage.map_coordinates

"""2014/03/17 moved here all routines from EA2019_align and from OP1S07_align."""

"""2014/03/06 from v1. Start from nanovea txt files use pure python to avoid coordinates mess. Assume data are saved with y fast axis and +,+ scanning
directions. It is not guaranteed to work with different scanning directions."""

"""angpos copiata qui, che non so come si fa ad importare in questo casino di directories.
CHANGES: angpos e' stata corretta funzionava con dati trasposti"""

## TRANSFORMATIONS
def crop_points(points,xrange=None,yrange=None,zrange=None,mask=False,poly=None,interactive=False):
    """crop a xyz points [Nx3], keeping only points inside xrange and yrange defined as (min,max).
        Interactive allows to set a region by zooming and/or a polygon by point and click
        (order of vertex matters)."""
    """experimental version, adds option boolean to return mask. useful e.g. to clean data based on crop on deltaR"""

    #outmask is sized as points and must remain same size all time.
    # at the end if flag mask is set, outmask is returned, else points are filtered.
    outmask=np.ones(points.shape[0]).astype(bool)
    #if mask is not None:
    #    outmask &= mask

    if interactive:
        """minimally tested."""

        curfig=plt.gcf() if plt.get_fignums() else None
        fig=plt.figure()
        plot_points(points,scatter=True)
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

    if poly:
        outmask=outmask & points_in_poly(points,poly)

    if xrange is not None:
        if xrange[0] is not None:
            outmask &= (xrange[0]<=points[:,0])
        if xrange[1] is not None:
            outmask &= (points[:,0]<=xrange[1])
    if yrange is not None:
        if yrange[0] is not None:
            outmask &= (yrange[0]<=points[:,1])
        if yrange[1] is not None:
            outmask &= (points[:,1]<=yrange[1])
    if zrange is not None:
        if zrange[0] is not None:
            outmask &= (zrange[0]<=points[:,2])
        if zrange[1] is not None:
            outmask &= (points[:,2]<=zrange[1])

    points=outmask if mask else points[outmask]

    return points


def rotate_points(points,theta,center=(0,0)):
    """returns rotated coordinates of 2D point(s) x ([Npoints x 2]) about a center with anticlockwise angle theta in rad. If 3D points are passed, z coordinate is maintained."""
    tx,ty=center
    points=np.array(points)
    if (np.shape(points)[-1]==3):
        return np.hstack([rotate_points(points[:,0:2],theta,center),points[:,2:]])
    else:
        if(np.shape(points)[-1]!=2):
            raise ValueError
    x,y=np.array(points[:,0]),np.array(points[:,1])
    cost=math.cos(theta)
    sint=math.sin(theta)
    x1=x*cost-y*sint + ty*sint - tx*(cost-1) #+cost*tx #
    y1=x*sint+y*cost - tx*sint - ty *(cost-1) #-sint*tx #
    return np.vstack((x1,y1)).T

def translate_points(points,offset=None):
    """returns translated coordinates. Useless routine, can be done with
    a simple sum points+offset"""
    
    ndim=points.shape[-1]
    
    if offset is not None:
        if len(offset) == ndim:
            return points + offset
        elif len(offset) == ndim - 1:
            return points[:,:-1] + offset

def _get_plane(points,pars=None,xrange=None,yrange=None,zrange=None,mask=None,returnPars=False):
    """
    Calculate the points of a plane from a set of 3D points.

    This function fits a plane to a set of 3D points and returns the points of the plane. 
    The plane is defined by the equation Ax + By + C = z, where A, B, and C are plane parameters.

    Parameters:
    points (ndarray): A numpy array of shape (n,3) representing n points in 3D space.
    pars (list or tuple, optional): A list or tuple of 3 elements [A, B, C] defining the plane. If None, the plane is calculated by best fit.
    xrange (tuple, optional): Range of x values (min, max) to include in the fit.
    yrange (tuple, optional): Range of y values (min, max) to include in the fit.
    zrange (tuple, optional): Range of z values (min, max) to include in the fit.
    mask (ndarray, optional): A boolean mask array to select points for fitting.
    returnPars (bool): If True, returns both the plane points and the parameters [A, B, C].

    Returns:
    ndarray: The points of the plane if returnPars is False. Otherwise, a tuple (plane_points, [A, B, C]).

    Raises:
    ValueError: If 'points' is not a 2D numpy array with shape (n,3).
    
    Examples:
    # Example 1: Using _get_plane with predefined parameters
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    pars = [1, 1, 0]
    plane = _get_plane(points, pars=pars)
    
        array([[0, 0, 0],
        [1, 1, 2],
        [2, 2, 4]])

    # Example 2: Using _get_plane to fit a plane on a set of points
    points = np.random.rand(100, 3)  # Random set of 100 points
    plane, parameters = _get_plane(points, returnPars=True)
    
        array([[0.37765124, 0.45062742, 0.5609915 ],
        ...
       [0.25846592, 0.48957061, 0.58587243]])

    """
    
    #2016/02/11 modified to return only plane (and not pars) by default.abs
    # It was previously returning a tuple (plane, pars) if and only if pars was not set.
    #   this breaks the interface, so I added the initial underscore in the function name, that it
    #   seems to be called only from inside points.py. If other programs are using the function
    #   they need to be adjusted by explicitly adding returnPars=True

    x,y,z=np.hsplit(points,3)
    if pars is None:
        if mask is not None:
            points=points[mask,:]
        pp=crop_points(points,xrange=xrange,yrange=yrange,zrange=zrange)
        pars=plane_fit(*np.hsplit(pp,3)).flatten()

    A,B,C=pars
    plane=np.hstack([x,y,A*x+B*y+C])
    if returnPars:
        return plane,pars
    else:
        return plane

def level_points(points,xrange=None,yrange=None,zrange=None,mask=None,pars=None,retall=False):
    """return the leveled points (after subtraction of best fit plane on selected points).
    If pars is provided as 3 elements vector is intended as plane coefficients and fit is not performed."""

    matrix=False
    if points.shape[1]!=3: #matrix input
        xgrid=np.arange(points.shape[0]) #irrelevant, they get lost when result is converted
        ygrid=np.arange(points.shape[1]) #   back to matrix.
        points=matrix_to_points(points,xgrid,ygrid)
        matrix=True  #flag for final resampling

    #get plane and pars, pars are used only if not passed (if they are passed there is no fit)
    plane=_get_plane(points,pars=pars,xrange=xrange,yrange=yrange,zrange=zrange,mask=mask,returnPars=True)
    #adjust for the number of variables needed.
    if pars is None:
        pars=plane[1]
        plane=plane[0]
    else:
        plane=plane[0]
    points=subtract_points(points,plane,resample=0)

    if matrix:
        points=resample_grid(points,xgrid,ygrid,matrix=True,resample=False)

    if retall:
        return points,pars
    else:
        return points
'''
def trim_points(points,interval,missing=None,invert=False):
    """Trim points keeping only points with z inside the interval.
    It differs from clip in the fact that points are eliminated, not replaced
    and clipping levels can be set only directly (as opposite as being calculated
    in automatic.
    But at the end is the same as crop. Use that."""

    aa=(points[:,2]<interval[1]) & (points[:,2]>interval[0]) #points to keep
    if invert:
       aa=np.logical_not(aa)
    na=np.logical_not(aa)   #negative of aa
    if missing is None:
        points=points[aa,:]
    else:
        points[na,2]=missing

    return points
'''


## ANALYSIS
def points_find_hull(pts):
    """return the convex hull non containing invalid (nan) points as a (np,2) array.
    hull=points_find_hull(pts)
    plt.plot(hull[:,0], hull[:,1], 'r--', lw=2)."""

    from scipy.spatial import ConvexHull
    
    pts = np.array(pts)
    if np.shape(pts)[1] == 2:
        xypts = pts
    elif np.shape(pts)[1] == 3:
        xypts=pts[~np.isnan(pts[:,2]),:2]
    else:
        raise ValueError("wrong shape for points in points_find_hull")

    cv = ConvexHull(xypts)

    ihull = cv.vertices
    # the vertices of the convex hull
    hull=xypts[ihull,:]

    return hull


def points_in_poly2(pts,vert):
    """return a boolen array, True if point is inside a polygon,
     given its vertices."""
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    inpoly=np.zeros(pts.shape[0]).astype(bool)

    point = Point(0.5, 0.5)
    polygon = Polygon(verts)
    return polygon.contains(point)

def points_in_poly(pts,verts):
    from matplotlib import path
    p=path.Path(verts)
    if pts.shape[1] == 3:
        pts=pts[:,:2]
    elif pts.shape[1] != 2:
        raise ValueError("wrong shape for points in points_find_hull")
    return p.contains_points(pts)
    #return np.hstack([pts[:,:2],p.contains_points(pts)[:,np.newaxis]])

def points_in_circle(pts, center, radius):
    """return a boolen array (mask), True if xy of point is inside a circle 
    (if center is given as two element vector), or sphere (if center is 3D)."""
    
    if np.size(center) == 2:
        mask = ((pts[:,:2]-np.array(center))**2).sum(axis=1)<=radius**2 
    else:
        raise NotImplementedError("3D center, this should filter only points inside a sphere.") 
    
    return mask

def clipStats(p,clip):
    print('clip for %s : %s'%(clip))
    print('z range: %s : %s'%(np.nanmin(p[:,2]),np.nanmax(p[:,2])))
    print('rms (clipped|unclipped): %s , %s'%(points_rms(p,clip=clip),points_rms(p)))
    print('number of elements clipped below|above %s,%s'%(p[np.where(p[:,2]<clip[0]),2].size,p[np.where(p[:,2]>clip[1]),2].size))
    print('xrange: %s,%s '%(np.nanmin(p[:,0]),np.nanmax(p[:,0])))
    print('yrange: %s,%s '%(np.nanmin(p[:,1]),np.nanmax(p[:,1])))

def points_rms(points,xrange=None,yrange=None,zrange=None,mask=None):
    """return the rms of the selected points."""

    points=crop_points(points,xrange=xrange,yrange=yrange,zrange=zrange,mask=mask)
    z=points[:,2]
    return np.sqrt(np.nanmean(z**2))

def histostats(points,bins=100,log=True,*args,**kwargs):
    """Plot histogram of z. Accept arguments for plt.hist.   """
    thr1=[-np.std(level_points(points)[:,2]),np.std(level_points(points)[:,2])]
    thr2=[-np.std(points[:,2]),np.std(points[:,2])]
    print('avg: %f dev:%f dev_lev:%f'%(np.mean(points[:,2]),thr1[1],thr2[1]))
    #%%timeit   #1 loops, best of 3: 19.1 s per loop
    #plt.figure(1)
    #plt.clf()
    #result=plt.hist(level_points(points)[:,2],bins=100,log=1)
    result=plt.hist(points[:,2],bins=bins,log=log,*args,**kwargs)
    #plot_points(points,shape=imsize,aspect='equal')
    plt.axvline(thr1[0],color='r')
    plt.axvline(thr1[1],color='r')
    plt.axvline(thr2[0],linestyle='-.',color='green')
    plt.axvline(thr2[1],linestyle='-.',color='green')
    #plt.title(plotTitle)
    #display(plt.gcf())
    return result

## I/O
def matrix_to_points2(mdata,x=None,y=None,xrange=None,yrange=None):
    ny,nx= mdata.shape  #changed 2015/11/04  to read ryan's data, this is because of python array shape (y,x) and because data are assumed to be in same order as image. Not sure gwyddion or nanovea data are in same format (but it should still work and giving a transposed result).
    #nx,ny= mdata.shape
    if x is None and y is None:
        if (xrange==None or yrange==None):
            #raise Exception
            x=np.arange(nx)
            y=np.arange(ny)
        else:
            x=np.linspace(*xrange,num=nx)
            y=np.linspace(*yrange,num=ny)

    xpoints,ypoints=[xy.flatten() for xy in np.array(np.meshgrid(x,y))]
    zpoints=mdata.flatten()
    points=np.vstack([xpoints,ypoints,zpoints]).T
    return points


def matrix_to_points(data,xgrid,ygrid,transpose=False):
    """this assumes that the order in data (after flattening) follows the order of
    x and y in meshgrid. If not (vertical direction first), set flag transpose.
    Not tested on arrays with different x and y, potentially a mess."""

    x,y=np.meshgrid(xgrid,ygrid)
    if transpose:
        data=data.T
    return np.vstack([x.flatten(),y.flatten(),data.T.flatten()]).T


def get_points(filename,x=None,y=None,xrange=None,yrange=None,matrix=False,addaxis=False,scale=None,center=None,skip_header=None,delimiter=',',*args,**kwargs):
    """
    Return a set of xyz points (N,3) from generic csv files in xyz or matrix format.

    For example for nanovea saved txt (matrix=False) or gwyddion saved matrix (matrix=True, xrange, yrange must be defined).

    `x` and `y` added 2018/02/17 make sense only for data in matrix format,
    allowing to ovveride or set values for the x and y axis and apply
    modifications that would not be straightforward on the returned value in
    `points` format.

    input x and y can be None or empty object (calculate from data size), M-element array (must fit data size) or range (2-el). If the last two cases (when appropriate axis are passed), they are always used and get priority on what is read from file, be careful to omit them unless you want to alter data.

    This way I can e.g. open a matrix file and return points with a modified x y grid. Setting x or y to an empty object rather than to None discards axis from file and use grid indices.

    addaxis: (if matrix is set) can be set to True to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_data.
    A single axis can be extracted if addaxis is set to string 'x', 'y'.
    `True` is equivalent to set to 'xy' (added 2020/06/18). 

    A complete description of the possible options is:
     read and use from file: addaxis=True, x= None
     read and discard from file, use passed: addaxis=True, x=np.array
     read and discard from file, use calculated: addaxis=True, x=[]
     don't read from file, use passed: addaxis=False, x=np.array
     don't read from file, use calculated: addaxis=False, x=None|[]

    center is the position of the center of the image in final coordinates (changed on 2016/08/10, it was '(before any scaling or rotation) in absolute coordinates.') If None coordinates are left unchanged.
        Set to (0,0) to center the coordinate system to the data.
    addaxis (if matrix is set) can be set to read values for axis in first row and column
        (e.g. if points were saved with default addaxis=True in save_points.
    2020/06/18 changes to logic in reading axis with matrix option, it was failing when x and y not set. Added possibility of individual `addAxis`
    TODO: reduce overlapping with `data2D.data_from_txt`.
    2018/02/17 reintroduced xrange even if discorauged. implemented x and y (unused as well) to axis, range
    or indices.

    """
    #import pdb
    #pdb.set_trace()
    if xrange is not None or yrange is not None:
        if verbose: 
            pdb.set_trace()
        print("WARNING: xrange and yrange options to get_points are obsolete, use x and y with two"+
               "elements to obtain same effect. Will be removed. I will correct for now")
        if x is None:
            x=xrange
        else:
            print("WARNING: xrange ignored, using X.")
        if y is None:
            y=yrange
        else:
            print("WARNING: yrange ignored, using Y.")

    #2014/04/29 added x and y as preferred arguments to xrange and yrange (to be removed).
    if skip_header is None:
        skip=0
    else:
        skip=skip_header

    mdata=np.genfromtxt(filename,skip_header=skip,delimiter=delimiter,*args,**kwargs)
    if (matrix):
        if addaxis == True: addaxis = 'xy'
        
        xx = None
        yy = None
        if addaxis == 'xy':
            yy,mdata=np.hsplit(mdata,[1])
            yy=yy[1:] #discard first corner value
            xx,mdata=np.vsplit(mdata,[1])
        elif addaxis == 'x':
            xx,mdata=np.vsplit(mdata,[1])
        elif addaxis == 'y':
            yy,mdata=np.hsplit(mdata,[1])
        elif addaxis != False:
            raise ValueError("Unrecognized value for addaxis (accepted values: `x`, `y`, `xy` or boolean. Passed value: "+str(addaxis))

        # x and y can be None or empty object(calculate from data size), M-element array (must fit data size)
        #  or range (2-el). Transform them in M-el vectors
        if x is None:
            # x override xx (first line, with addAxis) if both defined
            if xx is not None:
                x = xx
            else:
                x = np.arange(mdata.shape[1])
        else:
            if np.size(x) != mdata.shape[1]:
                if np.size(x) == 2:
                    x=np.linspace(x[0],x[1],mdata.shape[1])
                else:
                    raise ValueError("X is not correct size (or 2-el array)")            
        if y is None:
            # y override yy (first col, with addAxis) if both defined
            if yy is not None:
                y = yy
            else:
                y = np.arange(mdata.shape[0])
        else:
            if np.size(y) != mdata.shape[0]:
                if np.size(x) == 2:
                    y=np.linspace(y[0],y[1],mdata.shape[0])
                else:
                    raise ValueError("Y is not correct size (or 2-el array)")            
        points=matrix_to_points2(mdata,x,y)
    else:
        points= mdata

    #2016/08/10, moved scale before center because it was setting center then scaling also the center (e.g. changing sign). With this change, center is the final center.
    if scale is not None:
        if len(scale)==2:
            scale=np.append(scale,(1.0))
        points=points*scale

    if center is not None:
        if len(center)==2:
            center=np.append(center,(0.0))
        offset=np.hstack([((np.nanmax(points,axis=0)+np.nanmin(points,axis=0))/2)[0:2],0])-np.array(center)
        points=points-offset  #center on 0


    return points

def save_points(filename,points,xgrid=None,ygrid=None,shape=None,matrix=False,fill_value=np.nan,addaxis=True,**kwargs):
    """save points on a file. If matrix is true write in matrix form (in this case you have to
    provide the values for axis). Otherwise write as points in columns."""
    #2016/10/21 rewritten routine to use points_find_grid
    #2014/08/08 default fill_value modified to nan.
    #20140420 moved filename to second argument for consistency.

    #changed interface, set automatic correction
    if type(points)==str:
        import time
        if verbose: 
            pdb.set_trace()
        print("""WARNING: routine was modified to have filename as first argument, modify IMMEDIATELY
            the calling code. Corrected automatically for this time, but I will punish you waiting 5 seconds.""")
        time.sleep(5)
        filename,points=points,filename

    if os.path.splitext(filename)[-1]=='.fits' or matrix==True:
        #fix x and y according to input
        #calculate x and y from shape if provided, then overwrite if other options
        if shape is not None:
            assert (xgrid is None) and (ygrid is None)
            s=span(points,axis=0)
            xgrid=np.linspace(s[0,0],s[0,1],shape[0])
            ygrid=np.linspace(s[1,0],s[1,1],shape[1])
        #x and y are automatically calculated, then overwrite if xgrid and ygrid provided
        x,y=points_find_grid(points,result='grid')[1] #but I don't know how to add this to fits
        if xgrid is not None:
            x=xgrid
        if ygrid is not None:
            y=ygrid
        #resample points on a matrix

        data=resample_grid(points,matrix=True,xgrid=x,ygrid=y)
        #save output
        if os.path.splitext(filename)[-1]=='.fits':
            print("creating fits..")
            hdu=fits.PrimaryHDU(data)
            hdu.writeto(filename,overwrite=1,**kwargs)  #clobber=1,**kwargs)
            return
        else:
            if addaxis:
                #add first column and row with x and y coordinates, unless flag noaxis
                #is selected.
                data=np.vstack([x,data])
                data=np.hstack([np.concatenate([[np.nan],y])[:,None],data])
                points=data
            #if not, they are already in the correct format

    points = points.copy() #to avoid side effects
    points[np.isnan(points[:,2]),2]=fill_value
    
    np.savetxt(filename,points,**kwargs)

'''
def save_points(points,filename,xgrid=None,ygrid=None,shape=None,matrix=False,fill_value=np.nan,addaxis=True,**kwargs):
    """save points on a file. If matrix is true write in matrix form (in this case you have to
    provide the values for axis). Otherwise write as points in columns."""
    #2014/08/08 default fill_value modified to nan.
    #20140420 moved filename to second argument for consistency.
    if shape is None:
        shape=points_find_grid(points)[1]
    if matrix:
        if shape==None:
            assert len(xgrid.shape)==len(ygrid.shape)==1
        else:
            assert xgrid==ygrid==None
            x,y,z=np.hsplit(points,3)
            xgrid=np.linspace(x.min(),x.max(),shape[0])
            ygrid=np.linspace(y.min(),y.max(),shape[1])
        assert xgrid!=None
        assert ygrid!=None
        grid=np.vstack([g.flatten() for g in  np.meshgrid(xgrid,ygrid)]).T
        points=ip.griddata(points[:,0:2],points[:,2],grid,fill_value=fill_value,method=method)
        points=points.reshape(ygrid.size,xgrid.size)
        if addaxis:
            #add first column and row with x and y coordinates, unless flag noaxis
            #is selected.
            points=np.vstack([xgrid,points])
            points=np.hstack([np.concatenate([[np.nan],ygrid])[:,None],points])
        #points=np.hstack([grid,points[:,np.newaxis]])
    #if not, they are already in the correct format
    if os.path.splitext(filename)[-1]=='.fits':
        if matrix:
            print "creating fits.."
            hdu=fits.PrimaryHDU(points)
            hdu.writeto(filename,clobber=1,**kwargs)
            return
        else:
            print 'Fits can be saved only for matrix data.'
    else:
        np.savetxt(filename,points,**kwargs)

'''

#SHAPE AND RESAMPLING FUNCTIONS

def points_find_grid2(points,result='shape',sort='head',steps=None):
    """Given points as pointcloud, do some basic guess on shape and axis orientation
    of the grid.
    Works for raster points, even irregular and non rectangular, but not for scatter
    (step is estimated from the first two elements, it fails if elements are not sorted,
    points can be sorted xy or yx to avoid failure, like (xysort):
        a = a[a[:,1].argsort()]
        a = a[a[:,0].argsort(kind='mergesort')] #stable sort.
    Return a tuple (fastind,result), fastind is 1 for y and 0 for x,
    result can be 'shape', 'step'  or 'grid'.
    step can be provided as scalar or 2d vector to enforce step size

    sort can be None, 'xy', 'yx' (respectively try to guess, x or y faster)
        or 'none' (apply stable sort xy before calculating grid).

    This second version, works well in well-behaved cases, for more complex situations,
    better to check results (see critical cases below).

    """

    """N.B.: it fails on unsorted points:
    ff=r'C:\\Users\Vincenzo\Google Drive\Shared by Vincenzo\Metrology logs and data\Simulations\Coating_stress\MirrorStressIF\1mOAB_100x100\Mirror_Stress_IF_Iridium_C_total.csv'
    pf=get_points(ff,delimiter=' ')
    points_find_grid(pf)[1]
    gives (10498, 100)

    it fails on some data sets, e.g. PCO1.2S02_BCB_rep2 in POOL\pipeline\surfaceNB2
    """

    #2018/02/12 removed failed attempt of sorting points before result
    #in points_find_grid2
    #2018/09/29 readded at beginning according to docstring example

    # determine fastindex
    if sort == 'xy':
        fastind=0
    elif sort == 'yx':
        fastind=1
    elif sort == 'center':
        #try to guess
        #determines the fastest from the smallest step.
        # if steps are the same cannot determine. Use some point at the middle
        # rather than first point to reduce chance of error
        i = np.array([-1,0,1])+int(points.shape[0]/2)  #central index
        #three values of i are set to avoid it happens at at the end of a line
        d=[points[i+1,0]-points[i,0],points[i+1,1]-points[i,1]] #distances x and y from next point
        #d=[points[1,0]-points[0,0],points[1,1]-points[0,1]]
        if np.abs(min(d[0]))<np.abs(min(d[1])):
            fastind=0  #y in points
        elif np.abs(min(d[0]))>np.abs(min(d[1])):
            fastind=1   #x
        else:
            fastind=np.nan
    elif sort == 'head':   # determine fast index from the first points, this is same method as in the first implementation in points_find_grid (not 2)
        #determines the fastest from the smallest step.
        # if steps are the same cannot determine.
        d=[points[1,0]-points[0,0],points[1,1]-points[0,1]]
        if np.abs(d[0])<np.abs(d[1]):
            fastind=1  #y in points
        elif np.abs(d[0])>np.abs(d[1]):
            fastind=0   #x
        else:
            fastind=np.nan
    elif sort == 'none':
        #sort them as xy
        fastind=0
        points = points[points[:,1].argsort()]
        points = points[points[:,0].argsort(kind='mergesort')] #stable sort.
    else:
        raise ValueError ('sort must be "xy", "yx" or None, instead of %s '%sort)

    #determine nfast as the length of first change of sign in
    # calculate the shape of the matrix detecting
    # first change of sign of derivative of scanning
    # coordinate (end of line and return to first point)
    # in fast axis.
    sign=np.sign(d[fastind])
    #nfast=(np.sign(np.diff(points[:,fastind]))!=sign)[0]).sum()+1
    id=np.where(np.sign(np.diff(points[:,fastind]))!= sign)[0] #positions of change of sign, it's last point in line
    
    id=np.hstack([id,[points.shape[0]-1]])
    nfast=len(id)
    if len(id) == 1:
        # no change in slow coordinate, it is a single profile in fast coordinate
        print("""WARNING: single coordinate array with %i points detected."""%(points.shape[0]))
        nslow =  points.shape[0]
    else:
        nslow=np.max(np.diff(id)) #diff is the length of each line
        
    #nslow=points.shape[0]/nfast
    if nslow*nfast!=points.shape[0]:
        if verbose: 
            pdb.set_trace()
        print("""WARNING: points number [%i] doesn't match regular grid for size determined by points_find_grid [%i x %i],
        usually OK, but please double check results."""%(points.shape[0],nslow,nfast))

    # pdb.set_trace()
    if steps is not None:
        if steps==(0,0):
            steps=d #automatically determine
        raise NotImplementedError

    slowind=int(not(fastind))
    if result=='step':  #uses d
        if steps is None:
            steps=[d[fastind],d[fastind]]  #set all to fastind, then replace the slow one
            slowsteps=np.diff(points[id,slowind])
            if len(slowsteps) != 0:  # if one dim slowsteps is 1-element
                steps[slowind]=slowsteps[np.argmax(np.abs(slowsteps))]
            else:
                steps[slowind] = 0
        retval = steps

    elif result=='shape':  #uses nslow
        if steps is None:
            #here axis are switched meaning the position of index to
            # access the matrix, i.e. the matrix has shape (ny,nx)
            shape=[nslow,nslow]
            shape[slowind]=nfast
            retval=shape            
        else:
            retval = steps

    elif result=='grid':  #use points to calculate by points_find_grid
        #xs,ys,zs=np.hsplit(span(points,axis=0),3)
        xs,ys,zs=span(points,axis=0)
        # nx,ny=points_find_grid(points)[1]
        nx,ny=points_find_grid2(points, sort = 'head')[1]  # this replaces original call to points_find_grid
        xgrid=np.linspace(xs[0],xs[1],nx)
        ygrid=np.linspace(ys[0],ys[1],ny)
        retval=[xgrid,ygrid]
    else:
        raise ValueError

    return fastind,tuple(retval)

def points_find_grid(points,result='shape',sort=None):
    """Given points as pointcloud, do some basic guess on shape and axis orientation of the grid. Not many checks in this first version, you need to verify results.
    Works for raster points, even irregular and non rectangular, but not for scatter (step is estimated from the first two
    elements, it fails if elements are not sorted, points can be
    sorted xy or yx to avoid failure, like (xysort):
        a = a[a[:,1].argsort()]
        a = a[a[:,0].argsort(kind='mergesort')] #stable sort.
    Return a tuple (fastind,result), fastind is 1 for y and 0 for x,
    result can be 'shape', 'step'  or 'grid'."""

    """N.B.: it fails on unsorted points:
    ff=r'C:\\Users\Vincenzo\\Google Drive\Shared by Vincenzo\Metrology logs and data\Simulations\Coating_stress\MirrorStressIF\1mOAB_100x100\Mirror_Stress_IF_Iridium_C_total.csv'
    pf=get_points(ff,delimiter=' ')
    points_find_grid(pf)[1]
    gives (10498, 100)

    it fails on some data sets, e.g. PCO1.2S02_BCB_rep2 in POOL\pipeline\surfaceNB2"""

    #2018/02/12 removed failed attempt of sorting points before result
    #in points_find_grid2

    #determines the fastest from the smallest step.
    # if steps are the same cannot determine.
    d=[points[1,0]-points[0,0],points[1,1]-points[0,1]]
    if np.abs(d[0])<np.abs(d[1]):
        fastind=1  #y in points
    elif np.abs(d[0])>np.abs(d[1]):
        fastind=0   #x
    else:
        fastind=np.nan

    # calculate the shape of the matrix detecting
    # first change of sign of derivative of scanning
    # coordinate (end of line and return to first point)
    # in fast axis.
    sign=np.sign(d[fastind])
    #nfast=(np.sign(np.diff(points[:,fastind]))!=sign)[0]).sum()+1
    id=np.where(np.sign(np.diff(points[:,fastind]))!= sign)[0] #positions of changes of sign, list of last point of each line
    
    id=np.hstack([id,[points.shape[0]-1]])
    nfast=len(id)
    if len(id) == 1:
        # no change in slow coordinate, it is a single profile in fast coordinate
        print("""WARNING: single coordinate array with %i points detected."""%(points.shape[0]))
        nslow =  points.shape[0]
    else:
        nslow=np.max(np.diff(id)) #diff is the length of each line
    
    #nslow=points.shape[0]/nfast
    if nslow*nfast!=points.shape[0]:
        if verbose: 
            pdb.set_trace()
        print("""WARNING: points number [%i] doesn't match regular grid for size determined by points_find_grid [%i x %i],
        usually OK, but please double check results."""%(points.shape[0],nslow,nfast))

    slowind=int(not(fastind))
    if result=='step':
        steps=[d[fastind],d[fastind]]
        slowsteps=np.diff(points[id,slowind])
        if len(slowsteps) != 0:  # if one dim slowsteps is 1-element
            steps[slowind]=slowsteps[np.argmax(np.abs(slowsteps))]
        retval=steps
    elif result=='shape':
        #here axis are switched meaning the position of index to
        # access the matrix, i.e. the matrix has shape (ny,nx)
        shape=[nslow,nslow]
        shape[slowind]=nfast
        retval=shape
    elif result=='grid':
        #xs,ys,zs=np.hsplit(span(points,axis=0),3)
        xs,ys,zs=span(points,axis=0)
        nx,ny=points_find_grid(points)[1]
        xgrid=np.linspace(xs[0],xs[1],nx)
        ygrid=np.linspace(ys[0],ys[1],ny)
        retval=[xgrid,ygrid]
    else:
        raise ValueError

    return fastind,tuple(retval)

def rebin_points(tpoints,matrix=False,steps=None,*args,**kwargs):
    """Use `stats.binned_statistic_2d` to create a new surface from original points
    divided in bins centered on xy of final points. Points in a bin can be averaged
    (default) or binned statistics options can be used to obtain result values.
    
    It is theoretically the best way to create a surface on a grid with minimal loss of information,
    however it can be tricky, both for estimating errors (or values on empty bins) and for transformations (e.g. crop )."""
    
    
    x,y,z=list(zip(*tpoints))
    #T=kwargs.pop('transpose',False)
    xr=span(x)
    yr=span(y)

    if steps is not None:
        try:    #2-el
            if len(steps)==1:
                steps=[steps,steps]
            elif len(steps)!=2:
                raise ValueError ("1 or 2d only accepted.")
        except:  #1-el
            steps=[steps,steps]

        #this is necessary to keep step constant,
        #  note that #of points (inclusion of end) can vary according
        #  to floating point behavior of arange.
        xb=np.arange(*xr,step=steps[0])-steps[0]/2.
        xb=np.append(xb,xb[-1]+steps[0])
        yb=np.arange(*yr,step=steps[0])-steps[1]/2.
        yb=np.append(yb,yb[-1]+steps[1])
        bins=[xb,yb]

    else:
        bins=kwargs.pop('bins',None)

    ss=stats.binned_statistic_2d(x,y,z,bins=bins,*args,**kwargs)
    x2=np.array([(xx+yy)/2. for xx,yy in zip(ss[1][:-1],ss[1][1:])])
    y2=np.array([(xx+yy)/2. for xx,yy in zip(ss[2][:-1],ss[2][1:])])

    if matrix:
        return ss[0].T,x2,y2
    else:
        return matrix_to_points(ss[0],x2,y2)

"""
d=np.vstack([f.flatten() for f in np.meshgrid(arange(3),arange(3)+10.)]).T
d=vstack([d.T,d[:,0]**2+d[:,1]**2]).T
"""

## resample_points: resample points on other points and return points
## resample_grid: resample points on square grid. Return points or matrix (no x and y that is a bit confusing since they are optional argument)
## points_autoresample:  return data2D in form data,x,y automatically calculated

## crop problem: if I crop points and then do auto_resample, in general I can get a frame of nan.
##      Think to points not on a grid. When I crop the result is still not on a grid, so when I resample
##      edge points are lost. Possible solution is to interpolate first on a regular grid, then crop.
##      This is ok, but requires understanding from the user and most likely requires to use crop_data
##      in data2D. The argument cut was also introduced in resample_grid to overcome this problem
##      in a brutal way. Another possibility is to introduce crop interval as argument.

def resample_points(tpoints,positions,method = defaultmethod):
    """resample tpoints [Npoints x 3] on the points defined in positions [Mpoints x 2], or [Mpoints x 3]
    (in this case 3rd column is ignored).
    Return a [Nx x Ny , 3] vector of points. To get a (plottable) matrix of data use:
    plt.imshow(rpoints[:,2].reshape(xgrid.size,ygrid.size))."""
    assert tpoints.shape[1]==3
    z=ip.griddata(tpoints[:,0:2],tpoints[:,-1],positions[:,0:2],method=method)
    rpoints=np.hstack([positions[:,0:2],z[:,np.newaxis]])
    return rpoints

def resample_grid(tpoints,xgrid=None,ygrid=None,matrix=False,resample=True,method = defaultmethod):

    """resample tpoints [Npoints x 3] on the grid defined by two vectors xgrid [Nx] and ygrid [Ny].
    Return a [Nx * Ny , 3] vector of points, sorted in standard python order
    (x changes faster) or a matrix if matrix=True. if resample is set to False
    only x and y are changed and values are not touched (must maintain number of points and order).
    matrix=True-->points to matrix
    p=resample_grid(p) #straighten the grid of p changing data as little as possible
    matrix=False,resample=False-> Convert from matrix to points without resampling if matrix input?, useless if input is points (do two opposite operations that should cancel each other).

    """
    """ old (matrix=False):
    To get a (plottable) matrix of data use:
    plt.imshow(rpoints[:,2].reshape(ygrid.size,xgrid.size))."""

    assert tpoints.shape[1]==3
    if (xgrid is None) and (ygrid is None):
        xgrid,ygrid=points_find_grid(tpoints,'grid')[1]
    if resample: # 2022/10/20 if not resampling, you want to keep invalids
        tpoints=tpoints[np.isfinite(tpoints[:,2]),:]  #2018/10/01
    x,y=np.meshgrid(xgrid,ygrid)
    if resample:  #if both resample and matrix are False, two useless operations are performed and the final array is unchanged.
        z=ip.griddata(tpoints[:,0:2],tpoints[:,2],(x,y),method=method) #this is super slow, but still faster than the one in matplotlib
    else:
        z=tpoints[:,2].reshape(ygrid.size,xgrid.size)
    if matrix:
        return z
    else:
        return np.vstack([x.flatten(),y.flatten(),z.flatten()]).T
    #2015/08/31
    #xy=zip(*(x.flat for x in np.meshgrid(xgrid,ygrid)))
    #return np.vstack([np.array(xy).T,data.flatten()]).T #magical formula found by trial and error
    #2014/11/24 rpoints=np.vstack([x.T.flatten(),y.T.flatten(),z.T.flatten()]).T
    #return rpoints


def points_autoresample(points,cut=0,resample=True):
    """Use points_find_grid to determine the grid for points and resample.
    It should give minimal alteration of points when close to grid.
    This routine returns data,x,y
    Same result can be obtained by calling
    resample_grid (without providing x and y)
    AND points_find_grid to determine x and y."""

    '''
def points_autoresample(points,edge=0):
    """Edge is the number of points to cut at edges."""
    sh=points_find_grid(points)[1]
    st=points_find_grid(points,result='step')
    x=np.linspace(span(points,axis=0)[0][0],span(points,axis=0)[0][1],sh[0]) #do some cropping to cut out nan
    x=x[edge:len(x)-edge]
    y=np.linspace(span(points,axis=0)[1][0],span(points,axis=0)[1][1],sh[1])
    y=y[edge:len(y)-edge]
    '''
    #print 'obsolete, replace with resample_grid(points,matrix=True)'
    fastax,(x,y)=points_find_grid(points,result='grid')
    if cut>0:
        x=x[cut:-cut]
        y=y[cut:-cut]
    data=resample_grid(points,x,y,matrix=True,resample=resample)
    return data,x,y



def extract_profile(points,xy0,xy1=None,npoints=None,along=True,plot=False,method = defaultmethod):
    """extract a profile from xy0=(x0, y0) to xy1=(x1,y1).
    Return a couple of vectors x, y, z. The number of points can be set, otherwise is set
    accordingly to the longest profile dimension.
    If `along` is set (default), a two-dim x-z profile is returned with x distance
    along the profile from xy0.
    If called without second argument, extract point.
    #TODO, can be made faster by cropping data around profile.
    """
    
    # npoints not provided uses coordinates
    ## infer grid
    _,(xg,yg) = points_find_grid(points,result='grid') #complete grid in xg,yg

    if xy1 is None: #extract point
        if len(np.shape(xy0)) == 1:
            xy1 = xy0
        else:
            #iterate
            profiles = [extract_profile(points,p[0],p[1],npoints=npoints,
                along=along,plot=(i==0 & plot)) for i,p in enumerate(xy0)]
            if plot :
                ax1 = plt.gcf().axes[0]
                ax2 = plt.gcf().axes[-1]
                for p,prof in zip(xy0[1:],profiles[1:]):
                    #ax1 = plt.subplot(211,label='surf')
                    plt.sca(ax1)
                    p0,p1=p
                    plt.plot([p0[0],p1[0]],[p0[1],p1[1]])
                    #ax2 = plt.subplot(212,label='profs') #,sharex=ax1)       
                    plt.sca(ax2)
                    #pdb.set_trace()
                    r= prof[0]
                    z= prof[-1]
                    plt.plot(r,z)
            return profiles
            
    #span of extreme points
    xs = np.sort([xy0[0],xy1[0]])  
    ys = np.sort([xy0[1],xy1[1]])
    
    #  range for crop of useful data,
    ix = np.where((xg>=xs[0]) & (xg<=xs[1]))[0]
    iy = np.where((yg>=ys[0]) & (yg<=ys[1]))[0]
    #  add one points to each side
    if len(ix) == 0:
        ix=[(np.abs(xg - np.nanmean(xs))).argmin()]
    if ix[0] !=0: ix = np.insert(ix,0,ix[0]-1)
    if ix[-1] !=(len(xg)-1): ix = np.append(ix,ix[-1]+1)
    if len(iy) == 0:
        iy=[(np.abs(yg - np.nanmean(ys))).argmin()]
    if iy[0] !=0: iy = np.insert(iy,0,iy[0]-1)
    if iy[-1] !=(len(yg)-1): iy = np.append(iy,iy[-1]+1)
    
    if npoints is None:
        npoints=max(len(ix),len(iy))
    else:
        if xy0 == xy1:
            raise ValueError("npoints cannot be provided for single point")
    
    #adjusted range
    xr = span(xg[ix])
    yr = span(yg[iy])
    
    pp = crop_points(points,xr,yr)
    #points = crop_points(points,[xy0[0],xy1[0]],[xy0[1],xy1[1]])
    
    if np.all(xy0 == xy1):
        return ip.griddata(pp[:,0:2],pp[:,-1],xy0,method=method)
    
    xx=np.linspace(xy0[0],xy1[0],npoints)
    yy=np.linspace(xy0[1],xy1[1],npoints)
    r=np.sqrt((xx-xy0[0])**2+(yy-xy0[1])**2)

    z=ip.griddata(pp[:,0:2],pp[:,-1],np.vstack([xx,yy]).T,method=method)
    #pdb.set_trace()
    if along:
        #pp=np.vstack([r,z]).T
        pp=[r,z]
    else:      #pp=np.hstack([xx[:,np.newaxis],yy[:,np.newaxis],z[:,np.newaxis]])
        pp=[xx,yy,z]
        r=xx

    if plot:
        #plt.figure()
        ax1 = plt.subplot(211)
        plot_points(points,aspect='auto')
        plt.plot([xy0[0],xy1[0]],[xy0[1],xy1[1]],'r')
        xl=plt.xlim()
        ax2 = plt.subplot(212) #,sharex=ax1)
        if not along:
            ax1.get_shared_x_axes().join(ax1, ax2)
        plt.plot(r,z)
        #plt.xlim(*xl)
        plt.grid()
        #plt.colorbar().remove() #dirty trick to adjust size to other panels
        #this works better
        #pdb.set_trace()
        box1 = ax1.get_position()
        box2 = ax2.get_position()
        ax2.set_position([box1.x0, box2.y0, box1.width , box2.height])
        #pdb.set_trace()
    
    return pp #points[:,0],points[:,1]

'''
def plot_markers(m,subplots=0,points=None,w=None,**kwargs):
    """m is N points x,y {Nx2}. plot circles around points.
    if subplots is set, generate subplots with zoomed area."""

    if subplots and (points is not None):
        nrows = int(np.sqrt(len(m[:,0])))+1 #int(math.ceil(len(subsl) / 2.))
        plt.figure()
        fig, axs = plt.subplots(nrows, nrows,squeeze=True)
        for marker,ax in zip(m,axs.flatten()[0:len(m)]):
            plt.sca(ax)
            scatter=scatter if kwargs.has_key('scatter') else True
            plt.xlim(marker[0:1]+[-w/2,w/2])
            plt.ylim(marker[1:2]+[-w/2,w/2])
            plot_points(points,scatter=scatter,bar=0,**kwargs)
            # Make an axis for the colorbar on the right side
        for ax in axs.flatten()[len(m):]:
            plt.cla()
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        plt.colorbar()
    else:
        plt.plot(m[:,0],m[:,1],'o',markerfacecolor='none', c='w',lw=20,markersize=20)
'''

def plot_points(points,xgrid=None,ygrid=None,shape=None,units=None,resample=True,scatter=False,contours=0,bar=True,stats=True,
    **kwargs):
    #should be named show
    """resample xyz points [Nx3] to a grid whose axis xgrid and ygrid are given
    and plot it. If resample is set to False x and y positions are considered only for range,
    but they are not used to position the z values (it works if x and y are on an exact unrotated grid,
    resampling is slower, but exact).
    shape is in format (nx,ny) that is opposite to python convention.
    contours is contour spacing."""

    from pySurf.data2D import plot_data #here to avoid circular import


    #plot
    #plt.ioff()
    if np.size(units)==1:
        units=np.repeat(units,3)

    x,y,z=np.hsplit(points,3)
    #cmap=kwargs.pop('cmap','jet')
    aspect=kwargs.pop('aspect','equal') #needed because scatter plot don't have kw aspect
    #print id(points)
    #pdb.set_trace()
    plt.gca().set_aspect(aspect)
    if contours:
        scatter=False
        bar=False

    if scatter: #do scatterplot
        beamsize=kwargs.pop('s', 20) #I may use this to represent beamsize (round symbol) or lateral resolution (square)
        plt.scatter(x, y, c=z, s=beamsize, edgecolors='None', **kwargs)
        #plt.xlim(span(x))
        #plt.ylim(span(y))
    else:
        #if not scatter plot, grid can be provided as shape or as xgrid and ygrid axis. If not provided is automatically guessed.

        if xgrid is None and ygrid is None:
            if shape is None:
                shape=points_find_grid(points)[1]
            #xgrid=np.linspace(x.min(),x.max(),shape[0])
            #ygrid=np.linspace(y.min(),y.max(),shape[1])
            xgrid,ygrid=points_find_grid(points,'grid')[1]  #2018/10/10
        else:
            if not len(xgrid.shape)==len(ygrid.shape)==1:
                raise ValueError

        if resample:
            print("resampling...")
            z=resample_grid(points,xgrid,ygrid)[:,2]

        ## consider replacing all of the following with call to plot_data
        nx,ny=shape  #[xx.size for xx in (xgrid,ygrid)]
        xr,yr,zr=[span(xx) for xx in (xgrid,ygrid,z)]  #ranges
        #xxg,sx=np.linspace(xr[0],xr[1],nx,retstep=1)   #grids (not needed anywhere) and steps (needed only to adjust plot on pixel centers.
        #yyg,sy=np.linspace(yr[0],yr[1],ny,retstep=1)
        ##ranges for plot (intervals centered on xxg, yyg)
        #xr=xr+np.array((-sx/2,sx/2))
        #yr=yr+np.array((-sy/2,sy/2))

        xr = span_from_pixels(xr,nx)
        yr = span_from_pixels(yr,ny)

        z=z.reshape(ny,nx)

        if contours:
            levels=np.arange(zr[0],zr[1],contours)
            #CS=plt.contour(xgrid,ygrid,z,colors='b',extent=[xr[0],xr[1],yr[0],yr[1]],aspect=aspect,levels=levels,
            #    origin='lower', **kwargs)
            #version without extent
            CS=plt.contour(xgrid,ygrid,z,colors='b',aspect=aspect,levels=levels, \
                    origin='lower', **kwargs)
        else:
            plot_data(z,xgrid,ygrid, **kwargs)
            bar=False
            #plt.imshow(z,extent=[xr[0],xr[1],yr[0],yr[1]],interpolation='none',aspect=aspect,
            #origin='lower', **kwargs)
    
    plt.xlabel('X'+(" ("+units[0]+")" if units[0] is not None else ""))
    plt.ylabel('Y'+(" ("+units[1]+")" if units[1] is not None else ""))
    
    if stats:
        from plotting.captions import legendbox
        from pySurf.data2D import get_stats
        if scatter:
            xgrid,ygrid=x,y
        legendbox(get_stats(z,xgrid,ygrid,string=True))
    
    if bar:
        cb=plt.colorbar()
        if units[2] is not None:
            cb.ax.set_title(units[2])
    
    #plt.gca().autoscale(False) #2021/03/29, not sure what this was for, 
    #   it was giving wrong plot with scatter = True, removed.
    #plt.show()
    #plt.ion()
    
    return z
    

def subtract_points(p1,p2,xysecond=False,resample=True):
    """Subtract second set of points after interpolation on first set coordinates.
    If xySecond is set to True data are interpolate on xy of p2 and then subtracted."""

    #2018/02/15 moved later to save one copy in default case
    #p1=p1.copy()  #added 2015/01/13 together with resample option.
    #p2=p2.copy()  #this is unnecessary

    if resample:
        if not(xysecond):
            p2=resample_points(p2,p1)
        else:
            p1=resample_points(p1,p2)
            p2=p2.copy()
    else:
        p2=p2.copy()
    #at this point p1 and p2 are on same grid

    p2[:,2]=p1[:,2]-p2[:,2] #note that this is a slice, so array is modifyied

    return p2

def subtract_points2(p1,p2,xySecond=False,resample=True):
    """attempt to make subtract_points faster by transposing and acting on faster slice"""

    if resample:
        if not(xySecond):
            p2=resample_points(p2,p1)
        else:
            p1=resample_points(p1,p2)
            p2=p2.copy()
    else:
        p2=p2.copy()
    #at this point p1 and p2 are on same grid

    p2.T[2,:]=p1.T[2,:]-p2.T[2,:] #note that this is a slice, so array is modifyied

    return p2

def subtract_points_old(p1,p2,xySecond=False,resample=True):
    """Subtract second set of points after interpolation on first set coordinates.
    If xySecond is set to True data are interpolate on xy of p2 and then subtracted."""

    p1=p1.copy()  #added 2015/01/13 together with resample option.
    p2=p2.copy()  #this is unnecessary

    if resample:
        if not(xySecond):
            p2=resample_points(p2,p1)
        else:
            p1=resample_points(p1,p2)

    p1[:,2]=p1[:,2]-p2[:,2] #note that this is a slice, so array is modifyied

    return p1


#FUNCTIONS THAT CAN BE DONE WITH MASKS

def roicircle_points(points,radius,missing=None,invert=False):
    """Select points inside a circle of given radius. Points outside the circle are removed
    or replaced by a missing value."""
    #this is downsampling points and defining pts.

    aa=(points[:,0]**2+points[:,1]**2)<(radius**2)
    if invert:
       aa=np.logical_not(aa)
    na=np.logical_not(aa)   #negative of aa
    if missing is None:
        points=points[aa,:]
    else:
        points[na,2]=missing

    return points

"""
def clip_points(min=min,max=max,clipvalue=clipvalue,nsigma=nsigma,
        refimage=refimage,mask=mask):
    #TODO accept points as refimage
    #TODO if ADDMARKER is set a marker is added for clipped points

    clippeddata=clip( data, min=min,max=max,clipvalue=clipvalue,nsigma=nsigma,$
      refimage=refimage,mask=mask,torefimage=torefimage,$
      _strict_extra=extra)
"""

##DEVELOPMENT


def smooth_points(points,xywidth,xgrid=None,ygrid=None,shape=None,matrix=False):
    """resample points on a grid and perform moving average smoothin in x and y according to xywidth,
    if one component is None smoothing in that direction is not performed.
    Return points in usual coordinates, unless matrix flag is set."""
    data=points_to_matrix(points,xgrid,ygrid)
    if not(xywidth[0] is None):
        data=np.apply_along_axis(running_mean,0,data,xywidth[0])
    if not(xywidth[1] is None):
        data=np.apply_along_axis(running_mean,1,data,xywidth[1])
    if matrix:
        return data
    else:
        return matrix_to_points(data,xgrid,ygrid)


def smooth_points2(points,xywidth,xgrid=None,ygrid=None,shape=None):
    """NOT WORKING shape in format (nx,ny) as convention in points, opposite of python."""
    def gauss_kern(size, sizey=None):
        """ Returns a normalized 2D gauss kernel array for convolutions """
        size = int(size)
        if not sizey:
            sizey = size
        else:
            sizey = int(sizey)
        x, y = mgrid[-size:size+1, -sizey:sizey+1]
        g = exp(-(x**2/float(size)+y**2/float(sizey)))
        return g / g.sum()

    def blur_image(im, n, ny=None) :
        """ blurs the image by convolving with a gaussian kernel of typical
            size n. The optional keyword argument ny allows for a different
            size in the y direction.
        """
        g = gauss_kern(n, sizey=ny)
        improc = signal.convolve(im,g, mode='valid')
        return(improc)

    x,y,z=np.hsplit(points,3)
    if shape==None:
        if not len(xgrid.shape)==len(ygrid.shape)==1:
            return z    #skip plot
    else:
        assert xgrid==ygrid==None
        xgrid=np.linspace(x.min(),x.max(),shape[0])
        ygrid=np.linspace(y.min(),y.max(),shape[1])
    z=resample_grid(points,xgrid,ygrid)[:,2]
    nx,ny=[xx.size for xx in (xgrid,ygrid)]
    xr,yr=[span(xx) for xx in (xgrid,ygrid)]
    xxg,sx=np.linspace(xr[0],xr[1],nx,retstep=1)
    yyg,sy=np.linspace(yr[0],yr[1],ny,retstep=1)
    #ranges for plot (intervals centered on xxg, yyg)
    xr=xr+np.array((-sx/2,sx/2))
    yr=yr+np.array((-sy/2,sy/2))
    z=z.reshape(ny,nx)

    xwidth,ywidth=xywidth

    z=blur_image(z,xwidth,ywidth)
    plt.imshow(z,extent=[ygrid.min(),ygrid.max(),xgrid.max(),xgrid.min()],aspect=aspect,interpolation='none')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.colorbar()
    #plt.show()
    return z

if __name__=="__main__":
  pts=get_points(gfile, delimiter='')
  
  