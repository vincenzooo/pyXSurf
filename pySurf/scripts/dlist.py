

"""functions operating on a list of Data2D objects"""
# turned into a class derived from list 2020/01/16, no changes to interface,
# everything should work with no changes.

import os
import matplotlib.pyplot as plt
import numpy as np
import pdb

from pySurf.readers.format_reader import auto_reader
from pySurf.data2D import plot_data,get_data, level_data, save_data, rotate_data, remove_nan_frame, resample_data
from pySurf.data2D import read_data,sum_data, subtract_data, projection, crop_data, transpose_data, apply_transform, register_data
from plotting.multiplots import find_grid_size, compare_images, subplot_grid
from pySurf.psd2d import psd2d,plot_psd2d,psd2d_analysis,plot_rms_power,rms_power
from plotting.backends import maximize
from plotting.add_clickable_markers import add_clickable_markers2

from pySurf.points import matrix_to_points2
from copy import deepcopy
from dataIO.span import span
from dataIO.fn_add_subfix import fn_add_subfix
from pySurf.data2D import projection
from pySurf.data2D_class import Data2D
from pySurf.data2D import levellegendre
from pySurf.affine2D import find_rototrans,find_affine
from pySurf.readers.instrumentReader import fitsWFS_reader

from IPython.display import display
from dataIO.superlist import Superlist

## FUNCTIONS ##

# Functions operating on a dlist as Python list of Data2D objects.
# Can be used equally on a Dlist object.

# TODO:
# Move from other scripts:
from pySurf.scripts.repeatability import dcouples_plot
    
def topoints(data,level=None):
    """convert a dlist to single set of points containing all data.
    if level is passed, points are leveled and the value is passed as argument (e.g. level=(2,2) levels sag along two axis)."""
    if level is not None:
        plist = [d.level(level) for d in data] 
    plist = [d.topoints() for d in data]    
    return np.vstack(plist)

def load_dlist(rfiles,reader=None,*args,**kwargs):
    """Extracted from plot_repeat. Read a set of rfiles to a dlist.
    readers and additional arguments can be passed as scalars or lists.

    You can pass additional arguments to the reader in different ways:
     - pass them individually, they will be used for all readers
         load_dlist(.. ,option1='a',option2=1000)
     - to have individual reader parameters pass them as dictionaries (a same number as rfiles),
         load_dlist(.. ,{option1='a',option2=1000},{option1='b',option3='c'},..)
         in this case reader must be explicitly passed (None is acceptable value for auto).

    Example:
        dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
                units=['mm','mm','um'])

        dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
                'units':['mm','mm','um']},{'scale':(1,1,-1),
                'units':['mm','mm','um']},{'scale':(-1,-1,1),
                'units':['mm','mm','$\mu$m']}])
    2019/04/08 made function general from plot_repeat, moved to dlist.
    """

    #pdb.set_trace()
    if reader is None:
        #reader=auto_reader(rfiles[0])
        reader = [auto_reader(r) for r in rfiles]
        
    if np.size(reader) ==1:
        reader=[reader]*len(rfiles)
        
    '''
    if kwargs : #passed explicit parameters for all readers
        pdb.set_trace()
        kwargs=[kwargs]*len(rfiles)
    else:
        if np.size(args) ==1:
            kwargs=[args]*len(rfiles)
        else:
            if np.size(args) != len(rfiles):
                raise ValueError
    '''

    if kwargs : #passed explicit parameters for each reader
        # Note, there is ambiguity when rfiles and a kwargs value have same
        # number of elements()
        #pdb.set_trace()
        #vectorize all values
        for k,v in kwargs.items():
            if (np.size(v) == 1):
                kwargs[k]=[v]*len(rfiles)    
            elif (len(v) != len(rfiles)):
                kwargs[k]=[v]*len(rfiles)
            #else:  #non funziona perche' ovviamente anche chiamando esplicitamente, sara'
            #  sempre di lunghezza identica a rfiles.
            #    print ('WARNING: ambiguity detected, it is not possible to determine'+
            #    'if `%s` values are intended as n-element value or n values for each data.\n'+
            #    'To solve, call the function explicitly repeating the value.'%k)
    
    # 2020/07/10 args overwrite kwargs (try to avoid duplicates anyway).
    # args were ignored before.
    if not args:  #assume is correct number of elements
        args = []*len(rfiles)
    
    
    #pdb.set_trace()
    #transform vectorized kwargs in list of kwargs
    kwargs=[{k:v[i] for k,v in kwargs.items()} for i in np.arange(len(rfiles))]
    
    #kwargs here is a list of dictionaries {option:value}, matching the readers
    dlist=[Data2D(file=wf1,reader=r,**{**k, **a}) for wf1,r,k,a in zip(rfiles,reader,args,kwargs)]

    return dlist

def test_load_dlist(rfiles):

    dlist=load_dlist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
            units=['mm','mm','um'])

    dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
            'units':['mm','mm','um']},{'scale':(1,1,-1),
            'units':['mm','mm','um']},{'scale':(-1,-1,1),
            'units':['mm','mm','$\mu$m']}])
    return dlist,dlist2


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
    #datalist = [d() for d in datalist]]
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
        add_clickable_markers2(ax=a,propertyname=propertyname) #,block=True)

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

def add_markers(dlist):
    """interactively set markers, when ENTER is pressed,
    return markers as list of ndarray.
    It was align_active interactive, returning also trans, this returns only markers,
    transforms can be obtained by e.g. :
    m_trans=find_transform(m,mref) for m in m_arr]
      """

    #set_alignment_markers(tmp)
    xs,ys=find_grid_size(len(dlist),5)[::-1]

    # differently from plt.subplots, axes are returned as list,
    # beacause unused axes were removed (in future if convenient it could
    # make return None for empty axes)
    fig,axes=subplot_grid(len(dlist),(xs,ys),sharex='all',sharey='all')

    #axes=axes.flatten()  #not needed because a list
    maximize()
    for i,(d,ax) in enumerate(zip(dlist,axes)):
        plt.sca(ax)
        #ll=d.level(4,byline=True)
        ll = d
        ll.plot()
        #plt.clim(*remove_outliers(ll.data,nsigma=2,itmax=1,span=True))
        add_clickable_markers2(ax,hold=(i==(len(dlist)-1)))

    return [np.array(ax.markers) for ax in axes]

def align_interactive(dlist,find_transform=find_affine,mref=None):
    """plot a list of Data2D objects on common axis and allow to set
    markers. When ENTER is pressed, return markers and transformations"""

    m_arr = add_markers (dlist)

    #populate array of transforms
    mref = mref if mref is not None else m_arr[0]
    m_trans = [find_transform(m,mref) for m in m_arr]

    return m_arr,m_trans

def psd2an(dlist,wfun=np.hanning,
                ymax=None,outfolder="",subfix='_psd2d',*args,**kwargs):
    """2d psd analysis with threshold on sag removed data. Return a list of psd2d.
    ymax sets top of scale for rms right axis.
    if outfolder is provided, save psd2d_analysis plot with dlist names+subfix"""
    m_psd=[]
    title = kwargs.pop('title','')
    #pdb.set_trace()
    for dd in dlist:
        m_psd.append(dd.level(2,axis=0).psd(analysis=True,
            title=title,wfun=wfun,*args,**kwargs))
        #psd2d_analysis(*dd.level(2,byline=True)(),
        #                 title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs)
        #m_psd.append((fs,ps))
        ax=plt.gcf().axes[-1]
        ax.set_ylim([0,ymax])
        plt.grid(0)
        ax.grid()
        plt.suptitle(dd.name+' - hanning window - sag removed by line')
        if outfolder is not None:
            plt.savefig(os.path.join(outfolder,dd.name+subfix+'.png'))
        #pr=projection(pl[0].data,axis=1,span=1)
        #plot_psd(pl[0].y,pr[0],label='avg')
        #plot_psd(pl[0].y,pr[1],label='min')
        #plot_psd(pl[0].y,pr[2],label='max')
    return m_psd

def extract_psd(dlist,rmsthr=0.07,rmsrange=None,prange=None,ax2f=None,dis=False):
    """use psd2an, from surface files return linear psd."""
    m_psd=psd2an(dlist,rmsthr=rmsthr,rmsrange=rmsrange,prange=prange,ax2f=ax2f,dis=dis)

    plt.figure()
    plt.clf()
    labels=[d.name for d in dlist]
    m_tot=[]
    #gives error, m_psd is a list of PSD2D objects, there is no f,p, check psd2an
    for P,l in zip(m_psd,labels):
        (f,p) = P()
        ptot=projection(p,axis=1)
        #plot_psd(f,ptot,label=l,units=d.units)
        #np.savetxt(fn_add_subfix(P.name,'_calib_psd','.dat',
        #    pre=outfolder+os.path.sep),np.vstack([fs,ptot]).T,
        #    header='SpatialFrequency(%s^-1)\tPSD(%s^2 %s)'%(u[0],u[2],u[0]))
        m_tot.append((f,ptot))
    # plt.title(outname)

    plt.legend(loc=0)
    if dis:
        display(plt.gcf())
    return m_tot

def psd2d(dlist,ymax=None,subfix='_psd2d',*args,**kwargs):
    """2d psd analysis of a dlist. 
    Doesn't do any additional processing or plotting (must be done externally)
    [`outfolder` is being removed].
    Any parameter for `Data2D.psd` are accepted,
    however, parameters are not vectorized (must be the same for all data).
    [see example of vectorization e.g. in `load_dlist`] 
    Return a list of psd2d.
    
    from `psd2an`
    ymax sets top of scale for rms right axis.
    if outfolder is provided, save psd2d_analysis plot with dlist names+subfix"""
    
    m_psd=[]
    title = kwargs.pop('title','')
    #pdb.set_trace()
    for dd in dlist:
        m_psd.append(dd.psd(analysis=True,
            title=title,wfun=wfun,*args,**kwargs))
        #psd2d_analysis(*dd.level(2,byline=True)(),
        #                 title=os.path.basename(outfolder),wfun=wfun,*args,**kwargs)
        #m_psd.append((fs,ps))
        ax=plt.gcf().axes[-1]
        ax.set_ylim([0,ymax])
        plt.grid(0)
        ax.grid()
        plt.suptitle(dd.name+' - hanning window - sag removed by line')
        if outfolder is not None:
            plt.savefig(os.path.join(outfolder,dd.name+subfix+'.png'))
        #pr=projection(pl[0].data,axis=1,span=1)
        #plot_psd(pl[0].y,pr[0],label='avg')
        #plot_psd(pl[0].y,pr[1],label='min')
        #plot_psd(pl[0].y,pr[2],label='max')
    return m_psd

## CLASS ##

# Minimal implementation, should broadcast properties

class Dlist(Superlist):
    """A list of pySurf.Data2D objects on which unknown operations are performed serially."""           
    
    def topoints(self,level=True):
        """convert a dlist to single set of points containing all data."""
        plist = topoints(self.data,level = None)
        return plist  