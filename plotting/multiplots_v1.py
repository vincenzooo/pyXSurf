import math
import numpy as np
from matplotlib import pyplot as plt
from dataIO.span import span
from plotting.fignumber import fignumber


def smartcb(ax=None):
    """from different places online is a way to get colorbars same height than the plot."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if ax is None: ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    return plt.colorbar(cax=cax)


def find_grid_size(number, direction=0):
    """given a number of plots determine the grid size that better fits all plots.
    First number returned is biggest, it is up to the user to switch for use as rows/cols.
    Direction dictates which direction (0=horizontal, 1=vertical)"""
    s = int(math.sqrt(number))
    if (number-s**2) < ((s+1)**2-number):
        res = (s+1, s)
    else:
        res = (s+1, s+1)

    if direction == 1:
        res = res[::-1]

    return res


def subplot_grid(number,fignum=None,*args,**kwargs):
    """return a generator that plots n plots.
    Usage:
        for i,a in enumerate(subplot_grid(3)):
            a.plot(x,x**i) #or plt.plot

            also axes=[a.plot(x,x**i) for i,a in enumerate(subplot_grid(3))]"""

    gridsize = find_grid_size(number)

    fig = fignumber(fignum)

    """    if fignum==0:
        plt.clf()
    else:
        plt.figure(fignum)

    plt.clf()    """

    ax = None
    for i in range(number):
        ax = plt.subplot(gridsize[0], gridsize[1], i+1)
        yield ax


def compare_images(datalist, x=None, y=None, fignum=None, titles=None,
                   vmin=None, vmax=None, commonscale=False, direction=0,
                   *args, **kwargs):
    """return a generator that plots n images in a list in subplots with shared
    zoom and axes. datalist is a list of 2d data on same x and y.
    fignum window where to plot, if fignnum is 0 current figure is cleared,
    if None new figure is created.
    """
    # this was changed a couple of times in interface, for example in passing
    # ((data,x,y),...) rather than (data1,data2),x,y
    # it can be tricky in a general way if data don't have all same size,
    # at the moment it is assumed they have.
    # in any case x and y are used only to generate the extent,
    # that is then adapted to the size of data.

    gridsize = find_grid_size(len(datalist), direction=direction)

    fig = fignumber(fignum)

    plt.clf()
    ax = None

    # this is to set scale if not fixed
    d1std = [np.nanstd(data) for data in datalist]
    std = min(d1std)

    for i, data in enumerate(datalist):
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
        axim = plt.imshow(data, extent=np.hstack([span(x),span(y)]),
                          interpolation='None', aspect='auto',
                          vmin=kwargs.get('vmin', d1mean-s),
                          vmax=kwargs.get('vmax', d1mean+s),
                          *args, **kwargs)
        plt.colorbar()
        yield ax

def diff_images(data1,data2,x=None,y=None,fignum=None,titles=None,vmin=None,vmax=None,
    commonscale=False, direction=0, *args, **kwargs):
    """plots two data sets with common axis and their difference. Return the three axis.
    """

    fig=fignumber(fignum)

    plt.clf()
    ax=None

    #this is to set scale if not fixed
    d1std=[np.nanstd(data) for data in (data1,data2)]
    std=min(d1std)

    if x is None:
        x=np.arange(data1.shape[1])
    if y is None:
        y=np.arange(data1.shape[0])

    ax1=plt.subplot (131,sharex=ax,sharey=ax)
    if titles is not None:
        plt.title(titles[i])

    s=(std if commonscale else d1std[0])
    d1mean=np.nanmean(data1)
    axim=plt.imshow(data1,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect='auto',vmin=kwargs.get('vmin',d1mean-s),vmax=kwargs.get('vmax',d1mean+s), *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    ax2=plt.subplot (132,sharex=ax,sharey=ax)
    if titles is not None:
        print("titles is not none, it is ",titles)
        plt.title(titles[i])

    s=(std if commonscale else d1std[1])
    d2mean=np.nanmean(data2)
    axim=plt.imshow(data2,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect='auto',vmin=kwargs.get('vmin',d2mean-s),vmax=kwargs.get('vmax',d2mean+s), *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    ax3=plt.subplot (133,sharex=ax,sharey=ax)
    plt.title('Difference (2-1)')
    diff=data2-data1
    axim=plt.imshow(diff,extent=np.hstack([span(x),span(y)]),
        interpolation='None',aspect='auto', *args, **kwargs)
    plt.colorbar(fraction=0.046, pad=0.04)

    return ax1,ax2,ax3

def align_images(ax2,ax3):
    """make two subplots, allow to draw markers on them, exit at enter key and
    return markers."""
    from plotting.add_clickable_markers import add_clickable_markers2
    
    ax1,ax2=add_clickable_markers2(ax=ax2),add_clickable_markers2(ax=ax3,hold=True)
    return ax1.markers,ax2.markers


def associate_plots(ax1,ax2,on=0):
    """if on is None toggle, if it is 0 set off, if 1 set on. Not implemented, for now
    associate and keeps on until figure is closed.
    """

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
        print("offset on unzoomed data, rescaled to other axes (dxc)",dxc)
        nxl=(xc-xs*zoom+dxc,xc+xs*zoom+dxc)  #new limits
        print(nxl)
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
    """just print the changes of axes.
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
    from matplotlib.backend_bases import NavigationToolbar2

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

    #from plotting.multiplots import subplot_grid
    x=np.array([1,2,3])
    plt.figure(1)
    plt.clf()
    for i,a in enumerate(subplot_grid(3)):
        a.plot(x, x**i)

    plt.figure(2)
    plt.clf()
    for i,a in enumerate(subplot_grid(3)):
        plt.plot(x,x**i)

    plt.figure(3)
    plt.clf()
    axes=[a.plot(x,x**i) for i,a in enumerate(subplot_grid(3))]



if __name__=='__main__':
    for i in range(10,20):
        print(i, find_grid_size(i))