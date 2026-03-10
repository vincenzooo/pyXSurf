""" vedi anche from `mpl_toolkits.axes_grid1`."""


import math
import numpy as np
from matplotlib import pyplot as plt
from dataIO.span import span
from plotting.fignumber import fignumber
from pySurf.data2D import plot_data


def smartcb(ax=None):
    """from different places online. It is a way to get colorbars same height than the plot."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if ax is None: ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    return plt.colorbar(cax=cax)

def align_axis_x(ax, ax_target):
    """Make x-axis of `ax` aligned with `ax_target` in figure. Useful to align shared
    axis when colorbar messes up the alignment.
    
    from: https://stackoverflow.com/questions/46694889/matplotlib-sharex-with-colorbar-not-working
    see also:
    https://stackoverflow.com/questions/54473465/aligning-x-axis-with-sharex-using-subplots-and-colorbar-with-matplotlib
    https://stackoverflow.com/questions/44682146/align-subplot-with-colorbar
    """
    posn_old, posn_target = ax.get_position(), ax_target.get_position()
    ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])

def set_sync_shared (axes, setevent = True):
    """From a list of axis with shared x, align them independently of colorbars or other modifiers.
    
    This function was established starting from code in `psd2d.psd2d_analysis` for cases     in which matplotlib misbehaves, in place of the much simpler previous alternative
        `plt.colorbar().remove()`.
    All axes are synchronized to the first of the list, 
    setevent binds the resize to figure event, so that any future resize will not break the sychronization.
    In case there are other conflicting events functions defined, it might be convenient not to 
         activate the event handling by setting setevent = False, but this is not tested at all.
    
    2026/02/26 Note that this function seems to adjust the position of the plots, not the axis. Also it works for plots placed along vertical direction, but not for horizontal ones, at least in the current implementation.
    
    usage:
    
        set_sync_shared ([ax1, ax2, ax3])
        
    or: 
    
        resize = set_sync_shared ([ax1, ax2, ax3], setevent = False)
        # do some plotting
        resize()
        """
        
    #original code:
    
        # # replaced with more evolved interactive resize:
        # #  plt.colorbar().remove() #dirty trick to adjust size to other panels
        # def resize(event):
        #     box1 = ax1.get_position()
        #     box2 = ax2.get_position()
        #     box3 = ax3.get_position()
        #     ax2.set_position([box1.x0, box2.y0, box1.width , box2.height])
        #     ax2.set_adjustable("box",share=True)
        #     ax3.set_position([box1.x0, box3.y0, box1.width , box3.height])
        #     ax3.set_adjustable("box",share=True)

        # plt.tight_layout()

        # cid = fig.canvas.mpl_connect('draw_event', resize)
        # cid2 = fig.canvas.mpl_connect('resize_event', resize)

        # resize(None)
    
    def resize(event = None):
        boxes = [ax.get_position() for ax in axes]
        box1 = boxes[0]
        for ax, box in zip(axes[1:], boxes[1:]):
            ax.set_position([box1.x0, box.y0, box1.width , box.height])
            ax.set_adjustable("box",share=True)

    plt.tight_layout()
    
    fig = plt.gcf()
    cid = fig.canvas.mpl_connect('draw_event', resize)
    cid2 = fig.canvas.mpl_connect('resize_event', resize)

    resize(None)
    
    return resize


def align_axis_y(ax, ax_target):
    """Make y-axis of `ax` aligned with `ax_target` in figure"""
    posn_old, posn_target = ax.get_position(), ax_target.get_position()
    ax.set_position([posn_old.x0, posn_target.y0, posn_old.width, posn_target.height])

def commonscale(fig=None,axis='xy'):
    """set scale of all axes in a figure to the range that includes all axes data. Static command, needs to be called again if data are updated.
    
    2022/11/23 axis can be set to 'x', 'y', 'z' or any combination like 'xy'."""
    
    rx=[]   # ranges
    ry=[]
    rz=[]
    
    sax = [] #scalable axis
    if fig is None: fig=plt.gcf()
    for ax in fig.axes:
        if len(ax.images) + len(ax.lines) != 0:  #dirty way to exclude e.g. colorbars
            rx.append(ax.xaxis.get_data_interval())  #under the assumption it has both x and ylim
            ry.append(ax.yaxis.get_data_interval())
            rz.append(ax.images[0].get_clim())
            sax.append(ax)
    
    commonlim=[[min([r[0] for r in rx]),max([r[1] for r in rx])],
                [min([r[0] for r in ry]),max([r[1] for r in ry])],
                [min([r[0] for r in rz]),max([r[1] for r in rz])]]
    
    for ax in sax:
        if 'x' in axis: ax.set_xlim(*commonlim[0])
        if 'y' in axis: ax.set_ylim(*commonlim[1])
        if 'z' in axis: ax.images[0].set_clim(*commonlim[2])

    plt.draw()
    
def test_commonscale():
    from pySurf.data2D import make_surf_legendre
    from plotting.multiplots import commonscale

    # create two surfaces of different shapes
    a = make_surf_legendre(np.arange(3),np.arange(4),[1,2])
    b = make_surf_legendre(np.arange(5),np.arange(2),[0,0.5])

    # single image
    plt.figure()
    plt.imshow(a)
    commonscale()

    # double image
    plt.figure()
    plt.subplot(121)
    plt.imshow(a)
    plt.subplot(122)
    plt.imshow(b)
    commonscale()

    # double line plot
    plt.figure()
    plt.subplot(121)
    plt.plot([1,2,3],[4,5,6])
    plt.subplot(122)
    plt.plot([2,4,5],[2,5,8])
    commonscale()

    # single image w cbar
    plt.figure()
    plt.imshow(a)
    plt.colorbar()
    commonscale()

    # double image w cbar
    plt.figure()
    plt.subplot(121)
    plt.imshow(a)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(b)
    plt.colorbar()
    commonscale()
    
    # mixed test figure
    plt.figure()
    plt.subplot(121)
    plt.plot([1,2,3],[4,5,6])
    plt.subplot(122)
    plt.imshow(np.arange(12).reshape([4,3]))
    plt.colorbar()
    commonscale()
    
    fig=plt.gcf()
    l,i,cb = fig.axes
    for o in l,i,cb: 
        print(o.lines)
        print(o.get_xlim())
    
''' 
def xfind_grid_size(number, smax=0):
    """given a number of plots determine the grid size that better fits all plots.
    First number returned is biggest, it is up to the user to switch 
    for use as rows or cols. 
    for example 
    print(find_grid_size(10))  #gives 4,3
    print(find_grid_size(10,3))  #gives 4,3
    print(find_grid_size(9))  #gives 3,3
    print(find_grid_size(9,2))  #gives 5,2
    
    for i in range(11):print(i,find_grid_size(i))
    for i in range(11):print(i,find_grid_size(i,2))

    """
    s = int(math.sqrt(number))  # 

    if number == s**2:
        res = (s, s)
    elif number <= (s+1)*s:
        res = (s+1, s)
    else:
        res = (s+1, s+1)
    """
    if (number-s**2) < ((s+1)**2-number):
        res = (s, s) 
    else:
        res = (s+1, s)
    """
    
    if smax > 0:
        if s + 1 > smax:
            res = (1+((number-1)//smax) , smax)

    #if direction == 1:
    #    res = res[::-1]

    return res
'''
 
def find_grid_size(number, smax=0, square = True, fill = False):
    #upgraded version. under development
    """given a number of plots determine the grid size that better fits all plots.
    First number returned is smaller, meaning more rows than columns.
    It is up to the user to switch for use as rows or cols. 
    Square and fill define the ideal shape for the grid.
    If fill is set returns a shape with no empty axis,
    if square is set the final shape will be as close as possible to a square,
        if not, it will be as long as it can in one dimension (meaning 
        that if smax is not set it will be a single line).
        
    for example 
    print(find_grid_size(10))  #gives 4,3
    print(find_grid_size(10,3))  #gives 4,3
    print(find_grid_size(9))  #gives 3,3
    print(find_grid_size(9,2))  #gives 5,2
    print (find_grid_size(3))  #gives 2,2
    print(find_grid_size(3,2),square=False)  #gives 2,2
    
    for i in range(11):print(i,find_grid_size(i))
    for i in range(11):print(i,find_grid_size(i,2))

    """
    
    #smax is set and you want to go for square -> square, or smax whatever is smaller
    #smax is set and you want to go for wide -> 
    #smax 0 square:1
    #smax:0 square:0
    
    if smax == 0:
        smax=number
    elif smax < 0:
        raise ValueError 
        
    #set res (array of sizes)
    if fill:
        raise NotImplementedError
    else:
        if square:
            s = int(math.sqrt(number))  # minimum possible size    
            if number == s**2:   #if exact square is ok
                res = (s, s)
            elif number <= (s+1)*s:   #needs to add one line
                res = (s+1, s)
            else:
                res = (s+1, s+1)      #if just below square, needs bigger
        else:
            maxsize = max(smax,1+number//smax)
            minsize=(number-1)//maxsize+1
            maxsize=min(number,maxsize) #adjust if 1 row
            return (minsize,maxsize)
    
    if s + 1 > smax:    
        res = ( smax,1+((number-1)//smax) )

    #if direction == 1:
    #    res = res[::-1]

    return res
    

def subplot_grid(number,size=0,smax=0, square = True, fill = False,*args,**kwargs):
    
    """
    Create a set of n=`number` subplots in a figure, automatically 
    chosing the shape of the grid. Returns figure and list of axis.
    Note that empty subplots (e.g. the 9th in a grid of 3x3 subplots when `number=8`) are not created.
    
    size: if this is provided, is used as size of the subplot grid. 
    smax: used in `find_grid_size` to limit the extension along axis for the subplot grid. See `find_grid_size` help.
    num: if integer is passed, plot on the corresponding figure. if None is passed, create a new one. TBD: pass a figure obj.
    Additional *args and **kwargs are passed to `plt.subplots`.
    
    
    2020/05/14 fixed bug on subplots removal, updated doc from:
    return a generator that plots n plots. It has advantage wrt plt.subplots of 
    not creating empty axes. The idea of making a generator is weird.
    modified to fig,axes as equivalent to plt.subplots 2018/09/06.
    Usage:
        for i,a in enumerate(subplot_grid(3)):
            a.plot(x,x**i) #or plt.plot

            also axes=[a.plot(x,x**i) for i,a in enumerate(subplot_grid(3))]"""

    gridsize = find_grid_size(number, smax = smax, square = square, fill = fill) if size == 0 else size
    
    #fig = fignumber(fignum)

    """    if fignum==0:
        plt.clf()
    else:
        plt.figure(fignum)

    plt.clf()    """
    
    #note:
    #    #this doesn't work, if you use subplot the sharex, sharey interface is different and
    # axes references need to be passed as opposite to subplots, that uses e.g. sharex='all'
    # in that case however all axis are created at first, so I use a workaround and remove
    # excess axis.
    #fig,axes=plt.subplots(xs,ys,sharex='all',sharey='all',squeeze=True)
    #  where sharex seems to need axis reference instead of string, as a consequence also this fails:        
    """
    axes=[]
    if i==0:
        axes.append(plt.subplot(gridsize[0], gridsize[1], i+1,*args,**kwargs))
    else:
        axes.append(plt.subplot(gridsize[0], gridsize[1], i+1,*args,**kwargs))
        
    #equivalent one-liner:
    #axes = [plt.subplot(gridsize[0], gridsize[1], i+1,*args,**kwargs) for i in range(number)]
    """    
    num = kwargs.get('num',plt.gcf().number)
    fig,axes=plt.subplots(*gridsize,num=num,squeeze=True,*args,**kwargs)
    #pdb.set_trace()
    if np.size(axes) > 1:  # flag `squeeze` returns np array with removed 
        # empty dimensions if >1 element, otherwise scalar.
        # I want to have it as a list (single or multi elements) 
        axes=axes.flatten().tolist()
    else:
        axes = list(axes)
    
    # remove extra axis
    for i in np.arange(number,len(axes))[::-1]: #remove from ends to avoid out of range index
        #a in axes[(np.arange(len(axes))+1)>number]: 
        a=axes.pop(i)
        a.remove()
    
    return fig,axes

''' #moved to dlist
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
'''

# i prossimi due sono piuttosto simili, si dovrebbero uniformare in una versione per data, x, y che viene poi chiamato anche da Data2D e
# spostare in pacchetto pySurf. Il leveling non e' necessario, andrebbe rimosso
#   in caso livellando a mano prima di chiamare la funzione.

from dataIO.outliers import remove_outliers

def plot_difference(p1t,p4, trim = None, dis=False):
    """plots and return difference of two Data2D objects, return difference.
    All data are plane leveled before plots, a common color scale is set after excluding outliers. Leveled difference is returned.
    If trim is other than None, plots are adjusted on valid data x and y range,
    if Trim = True empty borders are removed also from difference data."""
    
    if trim is not None:
        if trim:
            p1t=p1t.remove_nan_frame()
            p4=p4.remove_nan_frame()
        xr=span(np.array([span(d.remove_nan_frame().x) for d in [p1t,p4]]))
        yr=span(np.array([span(d.remove_nan_frame().y) for d in [p1t,p4]]))
    else:
        xr=span(np.array([span(d.x) for d in [p1t,p4]]))
        yr=span(np.array([span(d.y) for d in [p1t,p4]]))
        
    plt.clf()
    
    ax1=plt.subplot(131)
    p1t.level((1,1)).plot()
    #plt.title('PZT + IrC')
    plt.clim(*remove_outliers(p1t.level().data,nsigma=2,itmax=3,span=1))
    plt.grid()


    ax2=plt.subplot(132,sharex=ax1,sharey=ax1)
    p4.level((1,1)).plot()
    #plt.title('PZT')
    plt.clim(*remove_outliers(p4.level().data,nsigma=2,itmax=3,span=1))
    plt.grid()
    
    ax3=plt.subplot(133,sharex=ax1,sharey=ax1)
    diff=(p1t-p4).level((1,1))
    diff.name='Difference 1-2'
    diff.plot()
    plt.clim(*remove_outliers(diff.level().data,nsigma=2,itmax=3,span=1))
    plt.grid()
    plt.xlim(*xr) #this adjust all plots to common scale
    plt.ylim(*yr)   
    
    if dis:
        display(plt.gcf())
    
    return diff
    
def diff_images(data1,data2,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False, direction=0, *args, **kwargs):
    """plots two data sets with common axis and their difference. Return the three axis.
    Colorbars are formatted to be same height as plot.
    2018/06/19 use data2D routines, allowing to add parameters (e.g. stats legend).
    """

    fig=fignumber(fignum)

    plt.clf() 
    
    aspect = kwargs.pop('aspect','auto')
    #this is to set scale if not fixed
    d1std=[np.nanstd(data) for data in (data1,data2)]
    std=min(d1std)

    if x is None:
        x=np.arange(data1.shape[1])
    if y is None:
        y=np.arange(data1.shape[0])

    ax1=plt.subplot (131)
    s=(std if commonscale else d1std[0])
    d1mean=np.nanmean(data1)
    plot_data(data1,x,y,aspect=aspect,vmin=kwargs.get('vmin',d1mean-s),
        vmax=kwargs.get('vmax',d1mean+s),*args, **kwargs)    
    
    ax2=plt.subplot (132,sharex=ax1,sharey=ax1)
    s=(std if commonscale else d1std[1])
    d2mean=np.nanmean(data2)
    plot_data(data2,x,y,aspect=aspect,vmin=kwargs.get('vmin',d2mean-s),
        vmax=kwargs.get('vmax',d2mean+s),*args, **kwargs) 

    ax3=plt.subplot (133,sharex=ax1,sharey=ax1)
    plt.title('Difference (2-1)')
    diff=data2-data1
    plot_data(diff,x,y,aspect=aspect,*args, **kwargs) 

    axes=[ax1,ax2,ax3]
    if titles is not None:
        for ax,tit in zip(axes,titles):
            if tit is not None:
                ax.set_title(tit)
        
    return axes    
    
def xdiff_images(data1,data2,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False, direction=0, *args, **kwargs):
    """plots two data sets with common axis and their difference. Return the three axis.
    Colorbars are formatted to be same height as plot.
    """

    fig=fignumber(fignum)

    plt.clf()
    ax=None
    
    aspect = kwargs.pop('aspect','auto')
    #this is to set scale if not fixed
    d1std=[np.nanstd(data) for data in (data1,data2)]
    std=min(d1std)

    if x is None:
        x=np.arange(data1.shape[1])
    if y is None:
        y=np.arange(data1.shape[0])

    ax1=plt.subplot (131,sharex=ax,sharey=ax)

    s=(std if commonscale else d1std[0])
    d1mean=np.nanmean(data1)
    axim=plt.imshow(data1,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect=aspect,vmin=kwargs.get('vmin',d1mean-s),vmax=kwargs.get('vmax',d1mean+s), *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    ax2=plt.subplot (132,sharex=ax,sharey=ax)

    s=(std if commonscale else d1std[1])
    d2mean=np.nanmean(data2)
    axim=plt.imshow(data2,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect=aspect,vmin=kwargs.get('vmin',d2mean-s),vmax=kwargs.get('vmax',d2mean+s), *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    ax3=plt.subplot (133,sharex=ax,sharey=ax)
    plt.title('Difference (2-1)')
    diff=data2-data1
    axim=plt.imshow(diff,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect=aspect, *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    axes=[ax1,ax2,ax3]
    if titles is not None:
        for ax,tit in zip(axes,titles):
            if tit is not None:
                ax.set_title(tit)
        
    return axes

def align_images(ax2,ax3):
    """make two subplots, allow to draw markers on them, exit at enter key and
    return markers."""
    from plotting.add_clickable_markers import add_clickable_markers2
    
    ax1,ax2=add_clickable_markers2(ax=ax2),add_clickable_markers2(ax=ax3,hold=True)
    return ax1.markers,ax2.markers

def associate_plots_axis(ax1,ax2,axis=None,on=0):
    """
    Incomplete attempt to add axis option.
    Associate axis of two different subplots `ax1`, `ax2`. 
    if `on` is None toggle, if it is 0 set off, if 1 set on. Not implemented, for now
    associate and keeps on until figure is closed.
    `axis` tells which axis are associated (0 = x, 1 =y, None = both).
    """

    if axis is None:
        axis = [0,1]  #activate both x and y-axis
    else:
        axis = [axis] # will be checked with `in`
        
    # Declare and register callbacks
    def on_xlims_change(axes):
        #print("updated xlims: ", axes.get_xlim())
        #axes.associated.set_autoscale_on(False)
        xold=axes.xold
        xl=axes.xaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        axes.xold=xl

        #print("stored value (xold), from get_view_interval (xl)=",xold,xl)
        zoom=(xl[1]-xl[0]) /(xold[1]-xold[0])
        #print("zzom",zoom)
        olim=axes.associated.get_xlim()
        #print("other axis lim (olim):",olim)
        xc=(olim[1]+olim[0])/2  #central point of other axis
        xs=(olim[1]-olim[0])/2  #half span
        #dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xl[1]-xl[0])
        dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xold[1]-xold[0])
        #print("offset on unzoomed data, rescaled to other axes (dxc)",dxc)
        nxl=(xc-xs*zoom+dxc,xc+xs*zoom+dxc)  #new limits
        #print(nxl)
        axes.associated.set_xlim(nxl,emit=False,auto=False)

    def on_ylims_change(axes):
        #print "updated ylims: ", axes.get_ylim()
        #axes.associated.set_autoscale_on(False)
        xold=axes.yold
        xl=axes.yaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        axes.yold=xl

        #print "stored value (yold), from get_view_interval (yl)=",xold,xl
        zoom=(xl[1]-xl[0]) /(xold[1]-xold[0])
        #print "zzom",zoom
        olim=axes.associated.get_ylim()
        #print "other axis lim (olim):",olim
        xc=(olim[1]+olim[0])/2  #central point of other axis
        xs=(olim[1]-olim[0])/2  #half span
        dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xold[1]-xold[0])
        #print "offset on unzoomed data, rescaled to other axes (dxc)",dxc
        #nxl=(xc+dxc,xc)  #newlimits
        nxl=(xc-xs*zoom+dxc,xc+xs*zoom+dxc)
        axes.associated.set_ylim(nxl,emit=False,auto=False)

    def ondraw(event):
        # print 'ondraw', event
        # these ax.limits can be stored and reused as-is for set_xlim/set_ylim later
        #print event -> it is matplotlib.backend_bases.DrawEvent object (?)
        ax1.xold=ax1.get_xlim()
        ax2.xold=ax2.get_xlim()
        ax1.yold=ax1.get_ylim()
        ax2.yold=ax2.get_ylim()
        #print "on draw axis: axes xlims", ax1.xold, ax2.xold


    fig=ax1.figure
    cid1 = fig.canvas.mpl_connect('draw_event', ondraw)
    if ax2.figure != fig:
        cid2 = ax2.figure.canvas.mpl_connect('draw_event', ondraw)
    
    setattr(ax2,'associated',ax1)
    setattr(ax2,'associated',ax1)
    
    if 0 in axis:
        setattr(ax1,'xold',ax1.get_xlim())
        setattr(ax2,'xold',ax2.get_xlim())
        ax1.callbacks.connect('xlim_changed', on_xlims_change)
        ax2.callbacks.connect('xlim_changed', on_xlims_change)        
        
    if 1 in axis:
        setattr(ax1,'yold',ax1.get_ylim())
        setattr(ax2,'yold',ax2.get_ylim())
        ax1.callbacks.connect('ylim_changed', on_ylims_change)
        ax2.callbacks.connect('ylim_changed', on_ylims_change)


def associate_plots(ax1,ax2,axis=None,on=0):
    """
    Associate axis of two different subplots `ax1`, `ax2`.
    This is a different mechanism than shared axis, in that here zoom and pans are kept synchronized, but axis are independent. If the two plots have same scales on both axis, it works like shared axis.
    
    Run `test_associate`, `test_associate_zoom` for examples.
    
    TODO: toggle association: note that relative alignment of centers is maintained, to be able to manually align center a toggle mechanism must be implemented:
    if `on` is None toggle, if it is 0 set off, if 1 set on. Not implemented, for now
    associate and keeps on until figure is closed.
    TODO: being able to associate only on x or y `axis` tells which axis are associated (0 = x, 1 =y, None = both). See attempt in `associate_plots_axis`
    TODO: when home is clicked, plots are messed up, need to click twice.
    
    """

    if axis is None:
        axis = [0,1] #activate both x and y-axis
        
    # Declare and register callbacks
    def on_xlims_change(axes):
        print("updated xlims: ", axes.get_xlim())
        #axes.associated.set_autoscale_on(False)
        xold=axes.xold
        xl=axes.xaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        axes.xold=xl

        print("stored value (xold), from get_view_interval (xl)=",xold,xl)
        zoom=(xl[1]-xl[0]) /(xold[1]-xold[0])
        print("zzom",zoom)
        olim=axes.associated.get_xlim()
        print("other axis lim (olim):",olim)
        xc=(olim[1]+olim[0])/2  #central point of other axis
        xs=(olim[1]-olim[0])/2  #half span
        #dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xl[1]-xl[0])
        dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xold[1]-xold[0])
        print("offset on unzoomed data, rescaled to other axes (dxc)\n\n",dxc)
        nxl=(xc-xs*zoom+dxc,xc+xs*zoom+dxc)  #new limits
        #print(nxl)
        axes.associated.set_xlim(nxl,emit=False,auto=False)

    def on_ylims_change(axes):
        print ("updated ylims: ", axes.get_ylim())
        #axes.associated.set_autoscale_on(False)
        xold=axes.yold
        xl=axes.yaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        axes.yold=xl

        print ("stored value (yold), from get_view_interval (yl)=",xold,xl)
        zoom=(xl[1]-xl[0]) /(xold[1]-xold[0])
        print ("zzom",zoom)
        olim=axes.associated.get_ylim()
        print ("other axis lim (olim):",olim)
        xc=(olim[1]+olim[0])/2  #central point of other axis
        xs=(olim[1]-olim[0])/2  #half span
        dxc=((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xold[1]-xold[0])
        print ("offset on unzoomed data, rescaled to other axes (dxc)\n\n",dxc)
        #nxl=(xc+dxc,xc)  #newlimits
        nxl=(xc-xs*zoom+dxc,xc+xs*zoom+dxc)
        axes.associated.set_ylim(nxl,emit=False,auto=False)

    def ondraw(event):
        # print 'ondraw', event
        # these ax.limits can be stored and reused as-is for set_xlim/set_ylim later
        #print event -> it is matplotlib.backend_bases.DrawEvent object (?)
        ax1.xold=ax1.get_xlim()
        ax2.xold=ax2.get_xlim()
        ax1.yold=ax1.get_ylim()
        ax2.yold=ax2.get_ylim()
        #print "on draw axis: axes xlims", ax1.xold, ax2.xold


    fig=ax1.figure
    cid1 = fig.canvas.mpl_connect('draw_event', ondraw)
    if ax2.figure != fig:
        cid2 = ax2.figure.canvas.mpl_connect('draw_event', ondraw)

    setattr(ax1,'associated',ax2)
    setattr(ax1,'xold',ax1.get_xlim())
    setattr(ax1,'yold',ax1.get_ylim())

    setattr(ax2,'associated',ax1)
    setattr(ax2,'xold',ax2.get_xlim())
    setattr(ax2,'yold',ax2.get_ylim())

    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax1.callbacks.connect('ylim_changed', on_ylims_change)
    ax2.callbacks.connect('xlim_changed', on_xlims_change)
    ax2.callbacks.connect('ylim_changed', on_ylims_change)

def associate_plots_mw(ax1,ax2):
    """if on is None toggle, if it is 0 set off, if 1 set on. Not implemented, for now
    associate and keeps on until figure is closed.
    Reduced functions for debugging.
    """

    # Declare and register callbacks
    def on_xlims_change(axes):

        # print "updated xlims: ", axes.get_xlim()
        # axes.associated.set_autoscale_on(False)
        xold = axes.xold
        xl = axes.xaxis.get_view_interval()   # range of plot in data coordinates, sorted lr
        axes.xold = xl

        zoom = (xl[1]-xl[0]) / (xold[1]-xold[0])
        olim = axes.associated.get_xlim()
        print("---------x limit change-------")
        print("on ax1\n" if axes == ax1 else "on ax2\n")
        print("old: (%5.3f,%5.3f)" % tuple(xold),
              "new:(%5.3f,%5.3f)" % tuple(xl),
              " zoom:%5.3f" % zoom)
        xc = (olim[1]+olim[0])/2  # central point of other axis
        xs = (olim[1]-olim[0])/2  # half span
        dxc = ((xl[0]+xl[1])-(xold[0]+xold[1]))/2*(olim[1]-olim[0])/(xold[1]-xold[0])
        # print "offset on unzoomed data, rescaled to other axes (dxc)",dxc
        nxl = (xc-xs*zoom+dxc, xc+xs*zoom+dxc)  # new limits
        print("new range other axes:(%5.3f,%5.3f)\n" % tuple(nxl))
        axes.associated.set_xlim(nxl, emit=False, auto=False)


    def ondraw(event):
        print('------ ondraw --------')
        # these ax.limits can be stored and reused as-is for set_xlim/set_ylim later
        ax1.xold = ax1.get_xlim()
        ax2.xold = ax2.get_xlim()
        print("axes xlims: (%5.3f,%5.3f)" % (ax1.xold),
              ",(%5.3f,%5.3f)" % (ax2.xold), "\n")

    fig = ax1.figure
    cid1 = fig.canvas.mpl_connect('draw_event', ondraw)
    if ax2.figure != fig:
        cid2 = ax2.figure.canvas.mpl_connect('draw_event', ondraw)

    setattr(ax1, 'associated', ax2)
    setattr(ax1, 'xold', ax1.get_xlim())
    setattr(ax2, 'associated', ax1)
    setattr(ax2, 'xold', ax2.get_xlim())

    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax2.callbacks.connect('xlim_changed', on_xlims_change)

def associate_plots_muc(ax1,ax2):
    """just print the changes of axes. Minimal use case for debugging.
    """

    def on_xlims_change(axes):
        xold=axes.xold
        xl=axes.xaxis.get_view_interval()   #range of plot in data coordinates, sorted lr
        axes.xold=xl
        olim=axes.associated.get_xlim()
        print("---------x limit change-------")
        print("on ax1\n" if axes==ax1 else "on ax2\n")
        print("old: (%5.3f,%5.3f)"%tuple(xold), "new:(%5.3f,%5.3f)"%tuple(xl))
        axes.associated.set_xlim((5,8),emit=False,auto=False)

    def ondraw(event):
        print('------ ondraw --------')
        # these ax.limits can be stored and reused as-is for set_xlim/set_ylim later
        ax1.xold=ax1.get_xlim()
        ax2.xold=ax2.get_xlim()
        print("axes xlims: (%5.3f,%5.3f)"%(ax1.xold),",(%5.3f,%5.3f)"%(ax2.xold),"\n")

    cid1 = ax1.figure.canvas.mpl_connect('draw_event', ondraw)
    cid2 = ax2.figure.canvas.mpl_connect('draw_event', ondraw)

    setattr(ax1,'associated',ax2)
    setattr(ax1,'xold',ax1.get_xlim())
    setattr(ax2,'associated',ax1)
    setattr(ax2,'xold',ax2.get_xlim())

    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax2.callbacks.connect('xlim_changed', on_xlims_change)

def associate_plots_tb2(ax1,ax2):
    """attempt to use an alternative mechanism for adjusting plot scales
        based on toolbar history. potentially lighter and more robust since someone
        else took care of details. need to keep under control side effects,
        for example must work on rescaling from code.
    """

    def ondraw(event):
        print('------ ondraw --------')
        # these ax.limits can be stored and reused as-is for set_xlim/set_ylim later
        ax1.xold=ax1.get_xlim()
        ax2.xold=ax2.get_xlim()
        if not ax1.figure.canvas.toolbar._views.home() is None:

            print(ax1.figure.canvas.toolbar._views.home())

            print(np.array([aa[:2] for aa in ax1.figure.canvas.toolbar._views.home()]))
            print(np.array(ax1.xold))
            if (np.array([aa[:2] for aa in ax1.figure.canvas.toolbar._views.home()])==np.array(ax1.xold)).all():
                print("home?!")
        print("axes xlims: (%5.3f,%5.3f)"%(ax1.xold),",(%5.3f,%5.3f)"%(ax2.xold),"\n")

    cid1 = ax1.figure.canvas.mpl_connect('draw_event', ondraw)
    cid2 = ax2.figure.canvas.mpl_connect('draw_event', ondraw)

    setattr(ax1,'associated',ax2)
    setattr(ax1,'xold',ax1.get_xlim())
    setattr(ax2,'associated',ax1)
    setattr(ax2,'xold',ax2.get_xlim())

    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    ax2.callbacks.connect('xlim_changed', on_xlims_change)

def set_home():
    from matplotlib.backend_bases import NavigationToolbar2
    home = NavigationToolbar2.home

    def new_home(self, *args, **kwargs):
        print('NEW HOME!')
        #print "method:",home

        home(self, *args, **kwargs)

    NavigationToolbar2.home = new_home
    return home

def restore_home(h):
    from matplotlib.backend_bases import NavigationToolbar2
    NavigationToolbar2.home=h

def test_associate():
    a=np.random.random(20).reshape((5,4))
    b=-a
    ax1=plt.subplot(121)
    plt.imshow(a,interpolation='none')
    ax2=plt.subplot(122)
    plt.imshow(b,interpolation='none')#,extent=[0,10,-1,1],aspect='equal')

    associate_plots(ax1,ax2,on=1)
    plt.show()
    return ax1,ax2

def test_associate_zoom():
    np.set_printoptions(3)
    a=np.random.random(20).reshape((5,4))
    b=a
    ax1=plt.subplot(121)
    plt.imshow(a,interpolation='none',extent=[0,20,-2,2],aspect='equal')
    ax2=plt.subplot(122)
    plt.imshow(b,interpolation='none',extent=[0,10,-1,1],aspect='equal')

    #associate_plots_muc(ax1,ax2)
    #associate_plots_tb2(ax1,ax2)
    associate_plots(ax1,ax2)
    associate_plots_mw(ax1,ax2)
    plt.show()
    return ax1,ax2

def test_subplot_grid():
    """test/example for subplot_grid"""
    
    plt.ion()
    
    #from plotting.multiplots import subplot_grid
    x=np.array([1,2,3])
    plt.figure(1)
    plt.clf()
    plt.suptitle('subplot_grid(3)')
    for i,a in enumerate(subplot_grid(3)[1]):
        a.set_title('x**%i'%i)
        a.plot(x, x**i)
    plt.tight_layout()

    plt.figure(2)
    plt.clf()
    for i,a in enumerate(subplot_grid(3)[1]):
        plt.plot(x,x**i)

    plt.figure(3)
    plt.clf()
    axes=[a.plot(x,x**i) for i,a in enumerate(subplot_grid(3)[1])]

    plt.figure(5)
    f,s=subplot_grid(8)
    for i,ss in enumerate(s):
        ss.plot(np.array([1,2,3])+i,[3,4,5])


if __name__=='__main__':
    for i in range(10,20):
        print(i, find_grid_size(i))