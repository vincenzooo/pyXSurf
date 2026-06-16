

"""functions operating on a list of Data2D objects"""
# turned into a class derived from list 2020/01/16, no changes to interface,
# everything should work with no changes.

## TODO: this module contains messy code from past analysis, especially in the psd section. The mechanism should be made consistent with Plist and Data2D and this code kept as legacy. Warnings to use updated methods should be added in all functions.
# e.g. PSDlist should be made of data2D_class.PSD2D objects.

import itertools
import os
import time
from copy import deepcopy
import warnings


import matplotlib.pyplot as plt
import numpy as np
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.span import span
from dataIO.superlist import Superlist
from IPython.display import display
from plotting.add_clickable_markers import add_clickable_markers2
from plotting.backends import maximize
from plotting.multiplots import commonscale, find_grid_size, subplot_grid
from pySurf.affine2D import find_affine
from pySurf.data2D import levellegendre, plot_data, projection
from pySurf.data2D_class import Data2D
from pySurf.readers.format_reader import auto_reader
from pySurf.readers.instrumentReader import fitsWFS_reader

## FUNCTIONS ##

# Functions operating on a dlist as Python list of Data2D objects.
# Can be used equally on a Dlist object.


    
def topoints(data,level=None):
    """convert a dlist to single set of points containing all data.
    if level is passed, points are leveled and the value is passed as argument (e.g. level=(2,2) levels sag along two axis)."""
    if level is not None:
        plist = [d.level(level) for d in data] 
    plist = [d.topoints() for d in data]    
    return np.vstack(plist)

def load_dlist(rfiles,reader=None,*args,**kwargs):
    """Extracted from `plot_repeat`. Read a set of `rfiles` to a `dlist`.
    readers and additional parameters (args or kwargs) can be passed as scalars or lists of same len as `rfiles`.

    example:
        dlist=load_dlist(rfiles,reader=reader,scale=(-1,-1,1),
            units=['mm','mm','um'])
    
        #This is new interface, modified from the previous (intended) interface 2024/03/06:
        dlist2=load_dlist(rfiles,reader,scale=[(-1,-1,1),(1,1,-1),(-1,-1,1)],
            units=[['mm','mm','um'],['mm','mm','um'],['mm','mm','$\mu$m']])
        # or, equivalently:
        # dlist2=load_dlist(rfiles,reader,**{'scale':[(-1,-1,1),(1,1,-1),(-1,-1,1)],
        #     'units':[['mm','mm','um'],['mm','mm','um'],['mm','mm','$\mu$m']]})
        
        # see test_load_dlist
    
    2019/04/08 made function general from plot_repeat, moved to dlist.
    
    TODO: implement vectorization of parameters directly in dlist or Superlist. 
    """

    # import pdb
    # pdb.set_trace()
    # if reader is None:
    #     #reader=auto_reader(rfiles[0])
    #     reader = [auto_reader(r) for r in rfiles]
        
    # if np.size(reader) ==1:
    #     reader=[reader]*len(rfiles)
    
    if reader is None:
        #reader=auto_reader(rfiles[0])
        reader = [auto_reader(r) for r in rfiles]
        
    if len(np.shape(reader)) == 0: #vectorize #np.size(reader) == 1:
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
    
    '''
    if kwargs : #passed explicit parameters for each reader
        # Note, there is ambiguity when rfiles and a kwargs value have same
        # number of elements()
        #pdb.set_trace()
        #vectorize all values
        for k,v in kwargs.items():
            if (np.size(v) == 1): 
                kwargs[k]=[v for dummy in rfiles]    
            #elif (len(v) != len(rfiles)):
            else:
                try:
                    kwargs[k]=[v.copy() for dummy in rfiles]
                except AttributeError:  #fails e.g. if tuple which don't have copy method
                    kwargs[k]=[v for dummy in rfiles]  
            # else:  #non funziona perche' ovviamente anche chiamando esplicitamente, sara'
            # # sempre di lunghezza identica a rfiles.
            #    print ('WARNING: ambiguity detected, it is not possible to determine'+
            #    'if `%s` values are intended as n-element value or n values for each data.\n'%k+
            #    'To solve, call the function explicitly repeating the value, es. `units=[['um','um','nm']]*3.`')
            # in realtà la forma "esplicità" fallisce al plot, mi fa temere che non funzioni anche se le immagini l'altra funziona. 
            
    
        # 2020/07/10 args overwrite kwargs (try to avoid duplicates anyway).
        # args were ignored before.
        
        #if not args:  #assume is correct number of elements
        #    args = [[]]*len(rfiles)   ## Non va fatto cosi'!! senno' duplica "by ref"
        
        #pdb.set_trace()
        
        #transform vectorized kwargs in list of kwargs
        kwargs=[{k:deepcopy(v[i]) for k,v in kwargs.items()} for i in np.arange(len(rfiles))]
    else:
        kwargs = [{} for dummy in rfiles] 
    '''    
    
    '''
    if args:
        for a in args:
            if (np.size(a) == 1):
                args=[[] for dummy in rfiles]    
            elif (len(a) != len(rfiles)):
                args=[args for dummy in rfiles]  
    else:
        args=[args for dummy in rfiles]
    '''
    
    
    from dataIO import dicts
    kwargs = dicts.vectorize(kwargs,len(rfiles),force_if_n=True)
    # kwargs = [{'scale': (-1, -1, 1),'units': ['mm', 'mm', 'um']},     #correct result of vectorization
    # {'scale': (1, 1, -1), 'units': ['mm', 'mm', 'um']},
    # {'scale': (-1, -1, 1), 'units': ['mm', 'mm', '$\\mu$m']}]
    
    from dataIO import arrays
    reader = arrays.vectorize(reader,len(rfiles))
    args = arrays.vectorize(args,len(rfiles))
        
    #kwargs here is a list of dictionaries {option:value}, matching the readers
    #dlist=[Data2D(file=wf1,reader=r,**{**k, **a}) for wf1,r,k,a in zip(rfiles,reader,args,kwargs)]
    # import pdb
    # pdb.set_trace()
    #dlist=Dlist([Data2D(file=wf1,reader=r,*a,**k) for wf1,r,a,k in zip(rfiles,reader,args,kwargs)])
    
    dlist=Dlist([Data2D(file=wf1,reader=r,*a,**k) for wf1,r,a,k in zip(rfiles,reader,args,kwargs)]) # OK
    
    # for wf1,r,a,k in zip(rfiles,reader,args,kwargs):
    #     print("load %s with reader %s and args: %s, keywords:"%(os.path.basename(wf1),r,a),k)
    #     print('-----------------------\n')
    # dlist = None
    
    
    return dlist

def test_load_dlist(rfiles,reader=fitsWFS_reader):

    dlist=load_dlist(rfiles,reader=reader,scale=(-1,-1,1),
            units=['mm','mm','um'])
    
    # what are possible vectorization? a list of args can be made of any number of elements,
    #   es.:
    #   var |     scalar  |   vector
    # nsigma|        3    |    [1,3,5]
    # clim  |   [0,100]   |    [[0,10],[1,8]]
    #
    # for args there is no key, and each arg can be single or multiple, the difference between the
    #   two cases depends on what is expected as input.
    # for kwargs must be the same, as {'var':scalar|vector,'nsigma':scalar|vector,...}.
    #
    # N.B.: the different form [{'scale':(-1,-1,1),
    #       'units':['mm','mm','um']},{'scale':(1,1,-1),
    #       'units':['mm','mm','um']}]
    #   cannot work, as this is not distinguishable from args.
    
    # dlist2=load_dlist(rfiles,reader,[{'scale':(-1,-1,1),
    #         'units':['mm','mm','um']},{'scale':(1,1,-1),
    #         'units':['mm','mm','um']},{'scale':(-1,-1,1),
    #         'units':['mm','mm','$\mu$m']}])
    # dlist2=load_dlist(rfiles,reader,{'scale':[(-1,-1,1),(1,1,-1),(-1,-1,1)],
    #         'units':[['mm','mm','um'],['mm','mm','um'],['mm','mm','$\mu$m']]})
    
    dlist2=load_dlist(rfiles,reader, scale = [(-1,-1,1),(1,1,-1),(-1,-1,1)],
            units=[['mm','mm','um'],['mm','mm','um'],['mm','mm','$\mu$m']])
    
    return dlist,dlist2

def plot_data_repeat(dlist,name="",num=None,*args,**kwargs):
    """"given a list of Data2D objects dlist, plots them as subplots on a grid with shared x and y scales in maximized window. 
    colorscale is independent for each subplot.
    returns stats.
    num is the figure number to plot on a specofic figure, other arguments are passed to plot.
    """

    res=[]
    #fig,axes=plt.subplots(1,len(dlist),num=1)
    xs,ys=find_grid_size(len(dlist),3,square=False)
    fig,axes=plt.subplots(xs,ys,num=num,clear=True)
    plt.close() #prevent from showing inline in notebook with %matplotlib inline
    axes=axes.flatten()
    maximize()
    
    for i,(ll,ax) in enumerate(zip(dlist,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        ll.plot(stats=True,*args,**kwargs)
        res.append(ll.std())
        #plt.clim([-3,3])
        try:
            plt.clim(*(np.nanmean(ll.data)+np.nanstd(ll.data)*np.array([-1,1])))
        except AttributeError:
            plt.ylim([ll.min(),ll.max()])
    plt.suptitle(name+' RAW (plane level)')
    for ax in axes[:len(dlist)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
        
    commonscale(plt.gcf())
        
    return res       

def dcouples_plot(dlist):  # level=True, dfunk = None,
    """calculate and plots rotating differences, data are supposed to be already aligned. 
    plots are generated on a grid, x and y axes are shared, color scale is automatic for each subplot.
    
    2023/04/11 removed `level` and `dfunk` parameters, data need to be manually leveled. Note, data are leveled by default (level=True).
    
    2022/11/23 added modification to vertical axis, now it works also with Plist. removed on 24 because it was wrong and setting to range of last plot. ``plotting.multiplots.commonscale`` can be called after the plot to uniform ranges. 
    - removed level, data can be level externally before or afterwards.
    
    """    
    
    dcouples=[c[1]-c[0] for c in list(itertools.combinations(dlist, 2))]
    
    # if level:
    #    dcouples=[d.level() for d in dcouples]
    
    # dcouples = []
    # for c in itertools.combinations(dlist, 2):    
    #     if dfunk is None:
    #         dfunk = c[0].__sub__
    #     else:
    #         dfunk = getattr(c[0],dfunk)
            
    #     dcouples.append(dfunk(c[1]))
            
    plt.clf()
    maximize()
    
    xs,ys=find_grid_size(len(dcouples),square=True)[::-1]
    fig,axes=plt.subplots(xs,ys)
    plt.close() #prevent from showing inline in notebook with %matplotlib inline
    if len(np.shape(axes))>1:
        axes=axes.flatten()
    elif len(np.shape(axes))==0:
        axes=[axes]
    maximize()
    for i,(ll,ax) in enumerate(zip(dcouples,axes)):
        plt.subplot(xs,ys,i+1,sharex=axes[0],sharey=axes[0])
        ll.plot()
    for ax in axes[:len(dcouples)-1:-1]:
        fig.delaxes(ax)
        #plt.pause(0.1)
    #plt.tight_layout()
    #commonscale()

    #return [d.std() for d in [diff21,diff31,diff32]]
    return dcouples

from plotting.fignumber import fignumber

''''''
def plot_datalist(datalist, *args, **kwargs):
    
    """attempt to build as an iterator, so it can be called in differente settings,
    like grid or multiples figures, also vectorizing arguments.
    It can be a good idea, but only partially implemented. It is needed to keep into account the
    size of the generator, e.g. to set figure or axis before the single plot is called, which partially 
    defeats the simplicity of the approach, even if the following code works well:
    
        from pySurf.scripts.dlist import plot_datalist
        from plotting.multiplots import subplot_grid


        plt.close('all')
        datalist = [d() for d in dl]
        
        n = len(datalist)
        a = plot_datalist(datalist)

        # on separate figures
        for i in range(n):
            try:
                plt.figure()
                next(a)
            except StopIteration:
                break
            
        '''
        # on a grid
        fig, grid = subplot_grid(n)
        for ax in grid:
            try:
                plt.sca(ax)
                next(a)
            except StopIteration:
                break
        '''
    """
        
    for d in datalist:
    
        data, x, y = d
        plot_data(data,x,y,*args,**kwargs)
        ax = plt.gca()
        
        yield ax



def compare_images(datalist, x=None, y=None, fignum=None, titles=None,
                   vmin=None, vmax=None, commonscale=False, axis=0, axmax=0,
                   *args, **kwargs):
    """return a generator that plots n images in a list in subplots with shared
    zoom and axes. datalist is a list of data in format (data, x, y).
    x and y provide plot range (temporarily, ideally want to be able to 
    plot same scale).
    fignum window where to plot, if fignnum is 0 current figure is cleared,
    if None new figure is created.
    axis defines the axis with larger number of plots (default ncols >= nrows)
    axmax maximum nr of plots along shorter dim.
    """
    #modified again 2018/06/05 to accept list of data,x,y triplets. x and y are 
    # accepted as range.
    # old docstring was:
    # return a generator that plots n images in a list in subplots with shared
    # zoom and axes. datalist is a list of 2d data on same x and y.
    # fignum window where to plot, if fignnum is 0 current figure is cleared,
    # if None new figure is created.
    
    
    # this was changed a couple of times in interface, for example in passing
    # ((data,x,y),...) rather than (data1,data2),x,y
    # it can be tricky in a general way if data don't have all same size,
    # at the moment it is assumed they have.
    # in any case x and y are used only to generate the extent,
    # that is then adapted to the size of data.

    gridsize = find_grid_size(len(datalist),axmax)
    if axis == 1: gridsize=gridsize[::-1]

    fig = fignumber(fignum)

    plt.clf()
    ax = None

    # this is to set scale if not fixed
    d1std = [np.nanstd(data[0]) for data in datalist]
    std = min(d1std)

    for i, d in enumerate(datalist):
        """adjust to possible input formats"""
        
        data, x, y = d
        
        if x is None:
            x = np.arange(data.shape[1])
        if y is None:
            y = np.arange(data.shape[0])
        ax = plt.subplot(gridsize[0], gridsize[1], i+1, sharex=ax, sharey=ax)
        if titles is not None:
            print("titles is not none, it is ", titles)
            plt.title(titles[i])
        # plt.xlabel('X (mm)')
        # plt.ylabel('Y (mm)')
        s = (std if commonscale else d1std[i])
        d1mean = np.nanmean(data)
        aspect=kwargs.pop('aspect',None)
        axim = plt.imshow(data, extent=np.hstack([span(x),span(y)]),
                          interpolation='none', aspect=aspect,
                          vmin=kwargs.get('vmin', d1mean-s),
                          vmax=kwargs.get('vmax', d1mean+s),
                          *args, **kwargs)
        plt.colorbar()
        
        
        yield ax

def multimarkers(datalist):
    """add clickable markers to a list of data using `add_clickable_markers2`. 
    Not sure it is updated, see also Dlist."""
    
    
    plt.figure()
    
    #difficile condividere gli assi visto che non possono essere passati
    s=subplot_grid(len(datalist))
    a=[add_clickable_markers2(s[0])]
    for ss in s[1:-1]:
        plot_data(datalist)
        a.append(add_clickable_markers2(ss))
    a.append(add_clickable_markers2(s[-1],hold=True))
    return a

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

from dataIO.functions import prep_kw


class Dlist(Superlist):
    """A list of pySurf.Data2D objects on which unknown operations are performed serially."""           
    
    '''  
    # how to handle self here? How are values read from files loaded in self?
    # es.: 
    # # BUG: return empty dlist

    # from pySurf.data2D_class import Data2D
    # ff = [os.path.join(datafolder,f) for f in files]
    # Dlist([Data2D(file=ff[0],reader=nid_reader)])
    
    # test 2024/02/21 (implemented auto_reader) OK
    # Dlist([Data2D(file=ff,reader=nid_reader) for ff in files]).plot(mode='grid')
    # Dlist(files,reader=nid_reader).plot(mode='grid')
    # Dlist(files).plot(mode='grid')
    
    '''
        
    def __init__(self, dlist, reader=None,*args,**kwargs):
        
        # Note that reader==None and vectorization are kept into account in load_dlist
        if all(isinstance(item, str) for item in dlist):
            dlist = load_dlist(dlist, reader = reader,*args,**kwargs)
            
        super().__init__(dlist,*args)
        
    def topoints(self,level=True):
        """convert a dlist to single set of points containing all data."""
        plist = topoints(self.data,level = None)
        return plist  
        
    def plot(self,mode='figures',*args,**kwargs):
        """
        modes:
        figures - plot each graph in a separate window
        grid - makes a grid of plots
               
        return a list of axis.
        
        
        N.B.: this is used also by Plist with an assignment.
        """
        
        if 'type' in kwargs:
            warnings.warn(
                "`type` keyword is deprecated, use `mode` instead. I have applied a delay to force you to update the code.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            time.sleep(5)
            
            if 'mode' in kwargs:
                raise ValueError("Both 'type' and 'mode' keywords are provided. Please use only 'mode'.")
            else:
                mode = kwargs.pop('type')
            
        
        if mode == 'figures': 
            axes = [plt.figure(**prep_kw(plt.figure,kwargs)) for dummy in self]
        elif mode == 'grid':
            from plotting.backends import maximize
            maximize()            
            axes = subplot_grid(len(self),*args,**kwargs)[1]
        elif mode == 'all':
            # overlap on same ax, useful for partial maps or profiles.
            #fig, ax = plt.subplots(**prep_kw(plt.subplots,kwargs))
            ax  = plt.gca()
            axes = [ax] * len(self) # three references to same axis 
        
        for ax,d in zip(axes,self):
            try:
                plt.sca(ax)
            except ValueError:
                print("WARNING: axes not existing.")
            d.plot(*args,**prep_kw(plt.plot,kwargs))
        #print(args)
        
        return axes

'''            
    def plot(self,mode='figures',*args,**kwargs):
        """
        modes:
        figures - plot each graph in a separate window
        grid - makes a grid of plots, parameters can be controlled with options as `multiplots.subplot_grid`
               
        return a list of axis.
        
        N.B.: this is used also by Plist with an assignment (Plist.plot = Dlist.plot).
        """
        
        from dataIO import arrays, dicts
        
        args = arrays.vectorize(args,len(self))
        kwargs = dicts.vectorize(kwargs,len(self))
        
        if mode == 'figures': 
            axes = [plt.figure(**prep_kw(plt.figure,kwargs)) for dummy in self]
        elif mode == 'grid':
            from plotting.backends import maximize
            maximize()            
            axes = subplot_grid(len(self),**prep_kw(subplot_grid,kwargs))[1]
        elif mode == 'all':
            # overlap on same ax, useful for partial maps or profiles.
            axes = [plt.figure()] * len(self) # three references to same axis 
        # import pdb
        # pdb.set_trace()
        for ax,d, aa, kk in zip(axes,self, *args,*kwargs):
            try:
                plt.sca(ax)
            except ValueError:
                print("WARNING: axes not existing.")
            d.plot(*args,**prep_kw(d.plot,*aa,**kk))
            #d.plot(*args,**kwargs)
        #print(args)
        
        return axes
'''