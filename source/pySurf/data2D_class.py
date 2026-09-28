import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from dataIO import outliers
from dataIO.fn_add_subfix import fn_add_subfix
from dataIO.functions import update_docstring
from dataIO.span import span
# from pyProfile.profile_class import Profile
from pySurf import data2D, points
from pySurf.affine2D import find_affine
from pySurf.data2D import apply_transform as apply_transform_data
from pySurf.data2D import (crop_data, data_from_txt, data_histostats,
                           get_stats, level_data, plot_data, projection,
                           read_data, register_data, resample_data, rebin_data,
                           rotate_data, save_data, slope_2D, subtract_data,
                           sum_data, transpose_data)
from .data2D import plot_slope_slice, plot_slope_2D
from .psd2d import avgpsd2d
from pySurf.points import matrix_to_points2, points_autoresample
from pySurf.psd2d import (plot_psd2d, plot_rms_power, psd2d, psd2d_analysis,
                          rms_power)
from pySurf.readers.format_reader import auto_reader

"""TODO: add decorators for methods returning a copy of self. Add to all of them a 'name' argument to set the name of the returned object. Maybe also a 'plot' flag to plot the result."""

"""
2018/06/07 v1.3
v1.2 was not convenient, switch back to same interface as 1.1:
don't modify self, return copy.

2018/06/06 v 1.2
After attempt to modify interface in a way that modify self, 
switch back to same interface as 1.1. It is inconvenient to have 
always to create a copy, in particular in interactive mode when same 
operation can be repeated multiple times you need to inizialize data 
every time (e.g. rotate).

2018/06/06 v1.1
methods are written to consistently return a copy without modifying self.
But this doesn't work, always need to assign, cannot chain (with property 
assignment, it works on methods) or apply to set of data in list.

Give some attention to inplace operators that can link to external data. 
At the moment when class is created from data, x, y these are assigned directly 
to the property resulting in a link to the original data. Some methods have 
inplace operations that will reflect on initial data, others don't."""

"""
programming notes:
-inplace methods-
A method acting on Data2D (e.g. level) can modify (o reassign to) self and 
return self or return a copy.

e.g. a method crop (or remove nancols) that return slices of data and x/y 
reassign to self.data/x/y views of original data.

A method that reassigns self.data=...
most of the time stays linked to original data.
If inplace changes (e.g. element-wise assignment) are then performed on the 
property, the change is reflected in the original data.
See notebook Python Programming Notes.

To remove this behavior, a copy of the data or view must be performed at some 
point (e.g. explicitly self.data.copy() or implicitly deepcopy(self).



-attempt to subclass ndarray-
def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None, info=None):
    # Create the ndarray instance of our type, given the usual
    # ndarray input arguments.  This will call the standard
    # ndarray constructor, but return an object of our type.
    # It also triggers a call to InfoArray.__array_finalize__
    obj = super(Data2D, subtype).__new__(subtype, shape, dtype,    #(InfoArray
                                            buffer, offset, strides,
                                            order)
    # set the new 'info' attribute to the value passed
    #obj.info = info
    # Finally, we must return the newly created object:
    return obj

def __array_finalize__(self, obj):
    # ``self`` is a new object resulting from
    # ndarray.__new__(InfoArray, ...), therefore it only has
    # attributes that the ndarray.__new__ constructor gave it -
    # i.e. those of a standard ndarray.
    #
    # We could have got to the ndarray.__new__ call in 3 ways:
    # From an explicit constructor - e.g. InfoArray():
    #    obj is None
    #    (we're in the middle of the InfoArray.__new__
    #    constructor, and self.info will be set when we return to
    #    InfoArray.__new__)
    if obj is None: return
    # From view casting - e.g arr.view(InfoArray):
    #    obj is arr
    #    (type(obj) can be InfoArray)
    # From new-from-template - e.g infoarr[:3]
    #    type(obj) is InfoArray
    #
    # Note that it is here, rather than in the __new__ method,
    # that we set the default value for 'info', because this
    # method sees all creation of default objects - with the
    # InfoArray.__new__ constructor, but also with
    # arr.view(InfoArray).
    #self.info = getattr(obj, 'info', None)
    # We do not need to return anything
"""


class Data2D(object):  # np.ndarrays
    """A class containing 2D data with x and y coordinates and methods for analysis.
    
        Can be initialized with data | data, x, y | file | file, x, y.
        if x and y are coordinates if match number of elements, 
            or used as range if two element. 
        If provided together with file, they override x and y 
            read from file.
        
        Function methods return a copy with new values and don't alter original
           object. Reassign to variable to use as modifier:
           e.g. a=a.level( ... )

        Args:
            data (2D array or string): 2D data or file name (suitable reader must 
            provided).
            
            x, y (array): coordinates or ranges.
            
            file (str): alternative way to provide a data file.
            
            units (str array): 3-element array with units symbols for `x`, `y`, `data`  
            
            reader (function): reader function (see `pySurf.readers.instrumentReader`).
            
            name (str): sets the name of created object.  

            *args, **kwargs: optional arguments for `pySurf.data2D.register_data`. 
                        
        """

    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super().__new__(subtype, shape, dtype,    
                                            #(InfoArray
                                                buffer, offset, strides,
                                                order)
        import pdb
        pdb.set_trace()
        # set the new 'info' attribute to the value passed
        #obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        #self.info = getattr(obj, 'info', None)
        # We do not need to return anything
    
    
    #def __repr__(self):
        #breakpoint()
        #pass
        #repr(self)
        #return self
    """
    
    def __init__(
        self,
        data=None,
        x=None,
        y=None,
        file=None,
        reader=None,
        units=None,
        name=None,
        *args,
        **kwargs
    ):


        # from pySurf.readers.instrumentReader import reader_dic

        # pdb.set_trace()

        if isinstance(data, str):
            print("first argument is string, use it as filename")
            file = data
            data = None
        # pdb.set_trace()
        self.file = file  # initialized to None if not provided
        if file is not None:
            assert data is None
            # store in xrange values for x if were passed
            xrange = span(x) if x is not None else None
            yrange = span(y) if y is not None else None

            # pdb.set_trace()
            if reader is None:
                reader = auto_reader(file)  # returns a reader
            # calling without arguments skips register, however skips also reader argumnets, temporarily arranged with pop in read_data to strip all arguments for
            data, x, y = read_data(file, reader, *args, **kwargs)
            # register data and pass the rest to reader
            # pdb.set_trace()

            if np.size(x) == data.shape[1]:
                self.x = x
            elif np.size(x) == 2:
                # y and yrange are identical
                print("WARNING: 2 element array provided for X, uses as range.")
                x = np.linspace(*xrange, data.shape[1])
            elif xrange is not None:
                print(
                    "wrong number of elements for x (must be 2 or xsize_data [%i]), it is instead %i"
                    % (np.size(data)[1], np.size(x))
                )
                raise ValueError

            if np.size(y) == data.shape[0]:
                self.y = y
            elif np.size(y) == 2:
                # y and yrange are identical
                print("WARNING: 2 element array provided for Y, uses as range.")
                x = np.linspace(*yrange, data.shape[0])
            elif yrange is not None:
                print(
                    "wrong number of elements for y (must be 2 or ysize_data [%i]), it is instead %i"
                    % (np.size(data)[0], np.size(y))
                )
                raise ValueError

            # set self.header to file header if implemented in reader, otherwise set to empty string""
            try:
                # kwargs['header']=True
                # self.header=reader(file,header=True,*args,**kwargs)
                self.header = reader(file, header=True, *args, **kwargs)
            except TypeError:  # unexpected keyword if header is not implemented
                self.header = ""
                # raise
        else:
            if data is not None:
                if len(data.shape) != 2:
                    # pdb.set_trace()
                    print(
                        "WARNING: data are not bidimensional, results can be unpredictable!"
                    )
                if x is None:
                    x = np.arange(data.shape[1])
                else:   # x as range
                    if np.size(x) == 2:
                        x = np.linspace(*x, data.shape[1])
                if y is None:
                    y = np.arange(data.shape[0])
                else:   # y as range
                    if np.size(y) == 2:
                        y = np.linspace(*y, data.shape[0])

                # if data is not None:
                # se read_data calls register, this
                data, x, y = register_data(data, x, y, *args, **kwargs)
                # goes indented.

        self.data, self.x, self.y = data, x, y

        if np.size(units) == 1:
            if units is not None:
                units=[units,units,units]
        self.units = units  # can be None, string, or 3-element array
        if name is not None:
            self.name = name
        elif file is not None:
            self.name = os.path.basename(file)
        else:
            self.name = ""
        # print(name)

    def __call__(self):
        return self.data, self.x, self.y

    def __add__(self, other, *args, **kwargs):
        return self.__class__(
            *sum_data(self(), other(), *args, **kwargs),
            units=self.units,
            name=self.name + " + " + other.name
        )

    def __mul__(self, scale, *args, **kwargs):
        res = self.copy()
        if len(np.shape(scale)) <= 1:  # multiply by scalar(s)
            if np.size(scale) == 3:
                res.x = scale[0] * res.x
                res.y = scale[1] * res.y
                res.data = scale[2] * res.data
            elif np.size(scale) == 1:
                res.data = res.data * scale
            else:
                raise ValueError("wrong number of elements in Data2D multiplication")
        elif len(np.shape(scale)) == 2:  # multiply by matrix
            # completely untested
            print("warning, multiplication between arrays was never tested")
            assert scale.shape == self.data.shape
            self.data = self.data * scale
        elif isinstance(scale, Data2D):
            tmp = scale.resample(self)
            res = self.copy()
            res.data = self.data * tmp.data
        else:
            raise ValueError("Multiply Data2D by wrong format!")
        return res

    def __rmul__(self, scale, *args, **kwargs):
        return self.__mul__(scale, *args, **kwargs)

    def __neg__(self):
        return self.__mul__(-1)

    def __sub__(self, other, *args, **kwargs):
        assert self.units == other.units
        res = self.__class__(*subtract_data(self(), other(), *args, **kwargs), units=self.units)
        res.name = self.name + " - " + other.name
        return res

    def __truediv__(self, other):
        return self * (1.0 / other)
        
    def __repr__(self):
    
        return '<%s "%s" at %s>'%(type(self),self.name,hex( id(self)))
        
        ''' NSee notes in profile_class'''

    def merge(self, other, topoints=False):
        """Return the merged data between a and b.
        If `topoints` is True points cloud data are returned.
        Gaps are brutally interpolated (unless `topoints` is set)."""

        if self.units != other.units:
            raise "incompatible units"

        p1 = self.topoints()
        p2 = other.topoints()
        res = np.vstack([p1, p2])
        if topoints:
            return res

        res = points_autoresample(res)
        return self.__class__(*res, units=self.units, name=self.name + " // " + other.name)

    def plot(self, title=None, *args, **kwargs):
        """plot using data2d.plot_data and setting automatically labels and colorscales.
        by default data are filtered at 3 sigma with 2 iterations for visualization, pass nsigma = None to include all data.
        Additional arguments are passed to plot. 
        2023/01/17 was returnign aximage modified to return axis."""

        units = self.units if self.units is not None else ["","",""]
        nsigma0 = 1  # default number of stddev for color scale
        # import pdb
        # pdb.set_trace()
        # to change the default behavior
        if "stats" in kwargs:
            stats = kwargs.pop("stats")
            fmt = kwargs.pop("fmt",None)
        else: 
            stats = [[0,1,3],[6],[6]]
            # format for legend labels (replace "stdev" with "rms")
            fmt = kwargs.pop("fmt", ['mean: %.3g '+units[2],  # f'mean: {units[2]}% ',
               'rms: %.3g '+units[2],
               'PV: %.3g '+units[2],
               'size: %i X %i'])

        # to change the default behavior
        nsigma = kwargs.pop("nsigma", nsigma0)
        m = self.data
        #pdb.set_trace()
        res = plot_data(
            self.data,
            self.x,
            self.y,
            units=units,
            fmt=fmt,
            stats=stats,
            nsigma=nsigma,
            *args,
            **kwargs
        )
        if title is None:
            if self.name is not None:
                title = self.name
        plt.title(title)
        return res

    plot = update_docstring(plot, plot_data)

    def load(self, filename, *args, **kwargs):
        """A simple file loader using data_from_txt"""
        self.data, self.x, self.y = data_from_txt(filename, *args, **kwargs)

    load = update_docstring(load, data_from_txt)

    def save(self, filename, *args, **kwargs):
        """Save data using data2d.save_data"""
        return save_data(filename, self.data, self.x, self.y, *args, **kwargs)

    save.__doc__ = save_data.__doc__

    from functools import update_wrapper

    # @update_wrapper(rotate_data)  #this doesn't work as I would like

    def rotate(self, angle, *args, **kwargs):
        """call data2D.rotate_data, which rotate array of an arbitrary angle in degrees in direction
        (from first to second axis)."""
        res = self.copy()
        # FIXME: rotation doesn't work without conversion to points.
        usepoints =  kwargs.pop("usepoints",True)
        res.data, res.x, res.y = rotate_data(
            self.data, self.x, self.y, angle, usepoints=usepoints, *args, **kwargs
        )
        return res

    rotate = update_docstring(rotate, rotate_data)

    def rot90(self, k=1, *args, **kwargs):
        """call data2D.rotate_data, which uses numpy.rot90 to rotate array of an integer multiple of 90 degrees in direction
        (from first to second axis)."""
        res = self.copy()
        res.data, res.x, res.y = rotate_data(*res(), k=k, *args, **kwargs)

        return res

    rot90 = update_docstring(rot90, rotate_data)

    def shift(self, xoffset=None, yoffset=None, zoffset=None):
        """Shift data of given offsets along one or more axes.
        `offsets` can be provided either as 1 (data offset), 2 (x,y) or 3 separate values, or as a single 2 (x,y) or 3 elements vector."""

        # 1,2,3
        # None, 2, 3
        # None, None, 3
        # None, 2, None
        # 1, None, None
        # 1, None, 3
        # 1, 2, None
        # [1,2,3], None, None
        # [1,2,3], ... fallisce se non none
        # [1,2], None, None
        # pdb.set_trace()
        offsets = [0, 0, 0]
        if yoffset is None and zoffset is None:
            # 1, None, None
            # [1,2,3], None, None
            # [1,2], None, None
            if xoffset is not None:
                offsets[: np.size(xoffset)] = np.array(xoffset)
        else:
            assert np.size(xoffset) == 1
            offsets[0] = 0 if xoffset is None else xoffset
            offsets[1] = 0 if yoffset is None else yoffset
            offsets[2] = 0 if zoffset is None else zoffset
            # [1,2,3], ... fallisce se non none
            # [1,2], ... fallisce se non none

        # 1,2,3
        # None, 2, 3
        # None, None, 3
        # None, 2, None
        # 1, None, 3
        # 1, 2, None
        
        res = self.copy()
        res.x = res.x + offsets[0]
        res.y = res.y + offsets[1]
        res.data = res.data + offsets[2]

        return res

    tv = [
        [1, 2, 3],
        [None, 2, 3],
        [None, None, 3],
        [None, 2, None],
        [1, None, None],
        [1, None, 3],
        [1, 2, None],
        [[1, 2, 3], None, None],
        [[1, 2, 3], 5, None],
        [[1, 2], None, None],
    ]

    def transpose(self):
        res = self.copy()
        res.data, res.x, res.y = transpose_data(self.data, self.x, self.y)
        return res

    transpose = update_docstring(transpose, transpose_data)

    def apply_transform(self, *args, **kwargs):
        res = self.copy()
        # pdb.set_trace()
        res.data, res.x, res.y = apply_transform_data(
            self.data, self.x, self.y, *args, **kwargs
        )
        return res

    apply_transform = update_docstring(apply_transform, apply_transform_data)

    def apply_to_data(self, func, *args, **kwargs):
        """apply a function from 2d array to 2d array to data."""
        res = self.copy()
        res.data = func(self.data, *args, **kwargs)
        return res

    def crop(self, *args, **kwargs):
        """crop data making use of function data2D.crop_data, where data,x,y are taken from a"""
        res = self.copy()
        res.data, res.x, res.y = crop_data(res.data, res.x, res.y, *args, **kwargs)
        return res

    crop = update_docstring(crop, crop_data)

    def level(self, *args, **kwargs):
        res = self.copy()
        res.data, res.x, res.y = level_data(self.data, self.x, self.y, *args, **kwargs)
        return res

    level = update_docstring(level, level_data)

    def resample(self, other, *args, **kwargs):
        """TODO, add option to pass x and y instead of other as an object (also in rebin)."""
        res = self.copy()
        if isinstance(other,Data2D):
            if self.units is not None and other.units is not None:
                if self.units != other.units:
                    raise ValueError(
                        "If units are defined they must match in Data2D resample."
                    )
            resampled = resample_data(res(), other(), *args, **kwargs)
        else:
            resampled = resample_data(res(), other, *args, **kwargs)
            
        res.data, res.x, res.y = resampled
        
        return res
    resample = update_docstring(resample, resample_data)
    
    def rebin(self, other, *args, **kwargs):
        """TODO, add option to pass x and y instead of other as an object, similarly to resample."""
        
        res = self.copy()
        if isinstance(other,Data2D):
            if self.units is not None and other.units is not None:
                if self.units != other.units:
                    raise ValueError(
                        "If units are defined they must match in Data2D resample."
                    )
        resampled = rebin_data(res(), other(), *args, **kwargs)
            
        res.data, res.x, res.y = resampled
        
        return res
    resample = update_docstring(rebin, rebin_data)

    def divide_and_crop(self, n, m):
        """Divide data in n x m equal size data. Data, returned as Dlsit, are ordered as coordinates."""
        
        xmin,xmax = span(self.x)
        ymin,ymax = span(self.y)
        
        # Width and height of each sub-rectangle
        x_step = (xmax - xmin) / n
        y_step = (ymax - ymin) / m
        
        dl = []
        # Nested loops to iterate over each sub-rectangle
        for j in range(m):
            for i in range(n):
                # Calculating the bounds for this sub-rectangle
                x_start = xmin + i * x_step
                x_end = xmin + (i + 1) * x_step
                y_start = ymax - (j + 1) * y_step
                y_end = ymax - j * y_step
                dd = self.crop((x_start, x_end), (y_start, y_end))
                dd.name = fn_add_subfix(dd.name, ' (%i,%i)'%(i,j))
                # Call the crop function with the calculated bounds
                dl.append(dd)
                
        return Dlist(dl)

    def add_markers(self, *args, **kwargs):
        #f = plt.figure()
        self.plot()
        ax = add_clickable_markers2(*args, **kwargs)
        markers = ax.markers
        #plt.close(f)
        return markers

    def psd(
        self,
        wfun=None,
        rmsnorm=True,
        norm=1,
        analysis=False,
        axis=1,
        subfix="",
        name=None,
        *args,
        **kwargs):
        """return a PSD2D object with 2D psd of self.
        If analysis is set True, `psd2d_analysis` function is called to generate plots.
        Parameters proper of this function are passed as args. 
        You need to pass also title, it generates output,
          this is subject to change, at the moment, pass empty string to generate plots
          or string to create output graphics.
        subfix and name are used to control the name of returned object.
        units are set in units of self because of the ambiguity mentioned in
        pySurf.psd2d.psd_units, and consistently with functions in `pySurf.psd2d`.
        """

        if axis == 0:
            self = self.transpose()  # transpose to have x as first axis
            # N.B.: psd2d works well with axis, but it was failing because the assignment of x to PSD2D
            # was not switched. e.g.: d.psd(axis=0) was failing on non square data.
            
        if analysis:
            title = kwargs.pop("title", (name if name is not None else ""))
            f, p = psd2d_analysis(
                self.data,
                self.x,
                self.y,
                wfun=wfun,
                norm=norm,
                rmsnorm=rmsnorm,
                title=title,
                units=self.units,
                *args,
                **kwargs
            )
        else:
            f, p = psd2d(
                self.data,
                self.x,
                self.y,
                wfun=wfun,
                norm=norm,
                rmsnorm=rmsnorm,
                *args,
                **kwargs
            )

        newname = name if name is not None else fn_add_subfix(self.name, subfix)
        return PSD2D(p, self.x, f, units=self.units, name=newname)

    psd = update_docstring(psd, psd2d)
    psd = update_docstring(psd, psd2d_analysis)

    def remove_nan_frame(self, *args, **kwargs):
        res = self.copy()
        res.data, res.x, res.y = data2D.remove_nan_frame(
            self.data, self.x, self.y, *args, **kwargs
        )
        return res

    remove_nan_frame = update_docstring(remove_nan_frame, data2D.remove_nan_frame)

    def topoints(self):
        """convenience function to get points using matrix_to_points2."""
        return matrix_to_points2(self.data, self.x, self.y)

    def std(self, axis=None):
        """return standard deviation of data excluding nans"""
        return np.nanstd(self.data, axis=axis)

    def copy(self):
        """copy.deepcopy should work well."""
        return deepcopy(self)

    def stats(self,*args,**kwargs):
        
        units = kwargs.pop('units',self.units)
        #breakpoint()
        return get_stats(self.data,self.x,self.y,units=units,*args,**kwargs)
    stats = update_docstring(stats, get_stats)

    def printstats(self, label=None, fmt="%3.2g"):
        if label is not None:
            print(label)
        s = ("%s PV: " + fmt + ", rms: " + fmt) % (
            self.name,
            span(self.data, size=True),
            np.nanstd(self.data),
        )
        print(s)
        return s

    def align_interactive(self, other, find_transform=find_affine, retall = False):
        """interactively set markers and align self to other.
        Alignment is performed using the transformation returned by
        find_transform(markers1,markers2) after markers are interactively set.
        Return aligned Data2D object.
        There is an experimental version for dlist in scripts."""
        from pySurf.scripts.dlist import add_markers

        m1, m2 = add_markers([self, other])
        trans = find_transform(m1, m2)
        
        if retall:
            return self.apply_transform(trans), (m1,m2), trans
        else:
            return self.apply_transform(trans)

    def remove_outliers(self, fill_value=np.nan, mask=False, *args, **kwargs):
        """use dataIO.remove_outliers to remove outliers from data. return a new Data2D object with outliers replaced by `fill_value`. If `mask` is set returns mask (easier than extracting it from returned object)."""
        res = self.copy()
        m = outliers.remove_outliers(res.data, *args, **kwargs)  # boolean mask
        # pdb.set_trace()
        if mask:
            return m
        res.data[~m] = fill_value
        return res

    remove_outliers = update_docstring(remove_outliers, outliers.remove_outliers)

    def extract_profile(self, *args, raw=False, **kwargs):
        """ Extract one or more profiles from start to end points. Return a `profile_class.Profile` object unless `raw` is True."""
        
        from pyProfile.profile_class import Profile
        # import pdb
        # pdb.set_trace()
        p = self.topoints()
        prof = points.extract_profile(p,  *args, **kwargs)
        if raw:
            return prof
        else:
            return Profile(*prof,units=[self.units[0],self.units[2]])

    extract_profile = update_docstring(extract_profile, points.extract_profile)
    
    def projection(self, axis = 0, *args, **kwargs):
        """avg, returns x and y. Can use data2D.projection keywords `span` and `expand` to return data ranges."""
        from pyProfile.profile_class import Profile
        u = self.units if self.units is not None else ["","",""]
        res = Profile(self.y, projection(self.data, axis=axis, *args,**kwargs), 
                      units = ([u[1], u[2]] if self.units is not None else None), 
                      name = " ".join([self.name,"avg along axis %i"%(axis)])) 
    
        # print(self)
        return res
    projection = update_docstring(projection, data2D.projection)

    def histostats(self, *args, **kwargs):
        res = data_histostats(
            self.data, self.x, self.y, units=self.units, *args, **kwargs
        )
        plt.title(self.name)
        return res

    histostats = update_docstring(histostats, data_histostats, "\n-------------\n")

    def slope(self, *args, **kwargs):
        # import pdb
        # pdb.set_trace()
        scale = kwargs.pop("scale", (1.0, 1.0, 1.0))
        if self.units is not None:
            if scale is None:
                # check if x and y in mm and z in um.
                if self.units[0] == self.units[1]:
                    if self.units[0] == "mm":
                        if self.units[0] == "mm" and self.units[2] == "um":
                            scale = (1.0, 1.0, 1000.0)
                else:
                    raise ValueError("x and y different units in slope calculation")
            u = [self.units[0], self.units[1], "arcsec"]
        else:
            scale = (1.0, 1.0, 1.0)
            u = ["", "", "arcsec"]

        say, sax = slope_2D(self.data, self.x, self.y, scale=scale, *args, **kwargs)

        return Data2D(
            *sax,
            units = u ,
            name=self.name + " xslope"
        ), Data2D(
            *say,
            units=u,
            name=self.name + " yslope"
        )

    slope = update_docstring(slope, slope_2D)

    def plot_slope(self, slice = False, scale = (1.,1.,1.) , *args, **kwargs):
        """Use `data2D.plot_slope_2D` and `.plot_slope_slice` to ploto 4-panel x and y slopes.
        
        Plot surface, x and y slope maps (rms profile if `slice` is set) and slope (or rms) distribution.
        Accept all keywords for `data2D.plot_slope_2D` and `.plot_slope_slice`."""
        
        # check for scale
        if self.units is not None:
            if len(set(self.units)) != 1 and len(set(scale)) == 1:
                # more than one unit, but same scaling
                print ("WARNING: units are different for axis, but scaling is uniform")
                print ("units: ", self.units)
                print ("scale: ", scale)

        if slice:
            plot_slope_slice(self.data, self.x, self.y, scale = scale, *args, **kwargs)
        else:
            plot_slope_2D(self.data, self.x, self.y, scale = scale,  *args, **kwargs)

    plot_slope = update_docstring(plot_slope, plot_slope_2D)
    plot_slope = update_docstring(plot_slope, plot_slope_slice)

    
    
class PSD2D(Data2D):
    """It is a type of data 2D with customized behavior and additional properties
    and methods.
    """

    def __init__(self, *args, **kwargs):
        '''super is called implicitly (?non vero)
        """needs to be initialized same way as Data2D"""
        #if a surface or a wdata,x,y are passed, these are interpreted as
        super().__init__(*args,**kwargs)
        '''
        super().__init__(*args, **kwargs)
    
    def plot(self,linear = False,*args, **kwargs):
        if linear:
            res = self.plot(*args, **kwargs)
        else:
            res = plot_psd2d(self.y,self.data,
                    self.x,
                    units=self.units,
                    *args,**kwargs)
        if self.name:
            plt.title (self.name) #(" - ".join([self.name, plt.gca().get_title()]))
        return res
    
    
    def avgpsd(self, *args, **kwargs):
        """avg, returns a PSD (linear) object. Can use data2D.projection keywords `span` and `expand` to return PSD ranges."""
        
        #return Profile(self.y, projection(self.data, axis=1, *args, **kwargs), units = [self.units[1], self.units[2]]) 
        res = super().projection(axis = 1, *args, **kwargs)
        from pyProfile.profile_class import PSD
        
        return PSD(res.x, res.y, units = res.units, name = self.name, *args, **kwargs)
    avgpsd = update_docstring(avgpsd, avgpsd2d)
    
    def rms_power(self, plot=False, rmsrange=None, *args, **kwargs):
        """Calculate rms slice power by integrating .
        If plot is set also plot the whole thing."""

        # pdb.set_trace()
        if plot:
            return plot_rms_power(
                self.y,
                self.data,
                self.x,
                units=self.units,
                rmsrange=rmsrange,
                *args,
                **kwargs
            )
        else:
            """this is obtained originally by calling rms_power, however the function deals with only scalar input for rms range.
            Part dealing with multiple ranges is in plot_rms_power, but should be moved to rms_power."""
            if rmsrange is not None:
                if np.size(rmsrange) != 2:
                    raise NotImplementedError

            return rms_power(self.y, self.data, rmsrange=rmsrange, *args, **kwargs)




def test_rot90():
    a = np.ones(250).reshape((25, 10))
    a[6:7, 6:9] = 3
    d = Data2D(a)

    plt.close("all")
    d.plot()
    plt.title("original")

    plt.figure()
    # return a,d
    c = d.rot90()
    c.plot()
    plt.title("rotated")

    plt.figure()
    # return a,d
    c = d.rot90(k=2)
    c.plot()
    plt.title("rotated k=2")

    plt.figure()
    # return a,d
    c = d.rot90(k=2, center=(10, 5))
    c.plot()
    plt.title("rotated k=2 about (10,5)")


def test_class_init(wfile=None, *args, **kwargs):
    """test init and plot"""

    from dataIO.fn_add_subfix import fn_add_subfix
    from pySurf.data2D import load_test_data

    # from pySurf.data2D import data, load_test

    d1, x1, y1 = load_test_data(wfile, *args, **kwargs)

    plt.figure(1)
    plt.clf()
    # plt.suptitle(relpath)
    plt.title("use plot_data function")
    plot_data(d1, x1, y1, aspect="equal")

    a = Data2D(d1, x1, y1)
    plt.figure(2)
    plt.clf()
    plt.title("From data")
    a.plot(aspect="equal")

    b = Data2D(file=wfile, ytox=220 / 1000.0, center=(0, 0))
    plt.figure(3)
    plt.clf()
    plt.title("from filename")
    b.plot(aspect="equal")

    b.save(
        os.path.join(
            outpath, os.path.basename(fn_add_subfix(relpath.as_posix(), "", ".txt"))
        ),
        makedirs=True,
    )
    b.remove_nan_frame()

    plt.figure()
    plt.title("removed nans")
    b.plot(aspect="equal")


if __name__ == "__main__":
    plt.close("all")
    test_class_init()
    test_rot90()
