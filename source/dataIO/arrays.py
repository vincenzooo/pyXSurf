import numpy as np
from dataIO.span import span
from itertools import product
import warnings
from copy import deepcopy

warnings.simplefilter("once")

from typing import Iterable, TypeVar

T = TypeVar("T")

def is_monotonic(seq: Iterable[T], *, strict: bool = False) -> bool:
    """
    Return True if `seq` is monotonic (non-decreasing or non-increasing).
    If `strict` is True, require it to be strictly monotonic (no equal neighbors).

    Notes:
    - Empty or single-element sequences are considered monotonic.
    - For strict=True, a single-element sequence is still True
      (there are no equal consecutive values to violate strictness).

    Examples
    --------
    >>> is_monotonic([1, 1, 2])               # non-decreasing
    True
    >>> is_monotonic([1, 1, 2], strict=True)  # equal neighbors => not strictly monotone
    False
    >>> is_monotonic([3, 2, 2])               # non-increasing
    True
    >>> is_monotonic([1, 3, 2])               # changes direction
    False
    >>> is_monotonic([], strict=True)
    True
    >>> is_monotonic([5], strict=True)
    True
    """
    
    it = iter(seq)
    try:
        prev = next(it)
    except StopIteration:
        return True  # empty

    direction = 0   # +1 for increasing, -1 for decreasing, 0 unknown yet

    for x in it:
        # Handle equality
        if x == prev:
            if strict:
                return False
            prev = x
            continue

        # Decide or check direction
        if x > prev:
            if direction == -1:
                return False
            direction = 1
        elif x < prev:
            if direction == 1:
                return False
            direction = -1
        else:
            # Non-comparable case (e.g., NaN with numbers): treat as not monotonic
            return False

        prev = x

    return True


import numpy as np

def table_of_test_points(
    x,
    ys,
    x_target,
    *,
    labels=None,
    title="",
    x_prec="actual",      # "target", "actual", "both"
    x_name="x",
    x_fmt="{:.2f}",
    y_fmt="{:.3f}",
    pad=2,                # spaces between columns
):
    """Create a table of extracted values from a series of vectors `ys` (with same x) at positions closest to `x_target` and format them as a table string."""
    
    x = np.asarray(x)
    xt = np.asarray(x_target)

    if x.size == 0:
        raise ValueError("`x` is empty.")
    if x_prec not in {"target", "actual", "both"}:
        raise ValueError('`x_prec` must be one of {"target","actual","both"}.')

    ys_arr = [np.asarray(y) for y in ys]
    n = x.size
    bad = [i for i, y in enumerate(ys_arr) if y.size != n]
    if bad:
        raise ValueError(f"All `ys` must have the same length as `x` ({n}). Bad indices: {bad}")

    if labels is None:
        labels = [f"y{i}" for i in range(len(ys_arr))]
    if len(labels) != len(ys_arr):
        raise ValueError(f"`labels` length ({len(labels)}) must match `ys` length ({len(ys_arr)}).")

    # Closest indices
    idx = np.array([int(np.argmin(np.abs(x - t))) for t in xt], dtype=int)

    # Build rows as strings first
    if x_prec == "both":
        header = [f"{x_name}_target", f"{x_name}_actual", *labels]
    else:
        header = [x_name, *labels]

    rows = [header]

    for i, t in enumerate(xt):
        j = idx[i]
        if x_prec == "target":
            x_cells = [x_fmt.format(float(t))]
        elif x_prec == "actual":
            x_cells = [x_fmt.format(float(x[j]))]
        else:  # both
            x_cells = [x_fmt.format(float(t)), x_fmt.format(float(x[j]))]

        y_cells = [y_fmt.format(float(y[j])) for y in ys_arr]
        rows.append([*x_cells, *y_cells])

    # Compute column widths (header + all rows)
    ncol = len(rows[0])
    widths = [max(len(r[c]) for r in rows) for c in range(ncol)]

    # Format: first col left, numeric cols right (looks like a table)
    def fmt_row(r):
        out = []
        for c, cell in enumerate(r):
            if c == 0:
                out.append(cell.ljust(widths[c]))
            else:
                out.append(cell.rjust(widths[c]))
        return (" " * pad).join(out)

    lines = []
    if title:
        lines.append(title)
        lines.append("-" * len(title))

    lines.append(fmt_row(rows[0]))
    for r in rows[1:]:
        lines.append(fmt_row(r))

    return "\n".join(lines)


def stats (data=None,units=None,string=False,fmt=None,vars=None):
    """ Return selected statistics on data as numerical array or list of strings (one for each stats).  
    
    vars is a list of indices that select the variables to be included, wrt a list (if called without `data` returns a format representation of the variables in the list):
    0 - mean
    1 - stddev
    2 - rms
    3 - PV
    4 - min
    5 - max
    6 - number of elements
    
    N.B.: (1) is intended as the rms of the deviation from the mean, while (2) is the root mean square of the signal as value (wrt to zero).
    Note that span doesn't exclude nan data, put flag to tune this option.
    
    string if set to True return stats as strings. In this case a string `units` can be used to add a postfix to statistics. A finer control can be obtained by passing in `fmt` a list of format strings for each var.
    e.g.: the default is obtained with:
    
            fmt = ['mean: %.5g'+units,
                   'StdDev: %.5g'+units,
                   'rms: %.5g'+units,
                   'PV: %.5g'+units,
                   'min: %.5g'+units,
                   'max: %.5g'+units,
                   'n:  %i']
    
    2021/06/30 added rms (different from standard dev, which is centered about mean).
    """

    if vars is None: vars = [0,1,2,3,4,5,6]
    
    if data is not None:
        # `st` stats to return are filtered based on vars
        data = np.array(data)
        st = [np.nanmean(data),np.nanstd(data),
              np.sqrt(np.nansum(data**2)/np.size(np.where(~np.isnan(data))[0])),
              span(data,size=True),*span(data),np.size(data)]
        if len(vars) > 0:
            st = [s for i,s in enumerate(st) if i in vars]
        else:
            st = []
    
    # convert to string
    #pdb.set_trace()
    if string or data is None:
        # `units` must be a single string and it is applied according to default fmt.   
        # `units` defaults to empty string, for different units must be incorporated in `fmt`. 
        if units is None or len(units)==0:   # if not empty adjust spacing  
            units = ""
        else:
            units = " " + units 
        if fmt is None or data is None or not(fmt):
            fmt = ['mean: %.5g'+units,
                'StdDev: %.5g'+units,
                'rms: %.5g'+units,
                'PV: %.5g'+units,
                'min: %.5g'+units,
                'max: %.5g'+units,
                'n:  %i']
            
            # filter `fmt` for indices in vars
            fmt = [f for i,f in enumerate(fmt) if i in vars]
            
            if data is None:
                warnings.warn ("`stats` called without data, print and returns default vars.")
                return fmt    
        elif np.size(fmt) == 1:  #??
            #if single string replicates to all variables
            fmt = np.repeat(fmt,len(st))
            if units:
                fmt = [f+units for f in fmt]
            
        #otherwise `fmt` and  is assumend already of same size as `st`
        
        st = [f%s for f,s in zip(fmt,st)]
        # 2021/06/30 here st and fmt are of same length. 
        # before today it was expecting complete format
        # st = [s for i,s in enumerate(st) if i in vars]  # filter st
    
    return st

def regular_grid(vector, precision = 8, average = False, step = False):
    
    """
    Calculates a regular grid based on the input vector.

    Parameters:
    vector (array-like): A 1D array of positions representing the grid points.
    precision (int, optional): The number of decimal places to consider when rounding the differences. Default is 8.
    average (bool, optional): If True, the step size is calculated as the mean of the differences; 
                               if False, the step size is calculated as the median of the differences. Default is False.
    step (bool, optional): If True, returns the calculated step size as a float; 
                           if False, returns a regular grid as a numpy array. Default is False.

    Returns:
    float or numpy.ndarray: The calculated step size if `step` is True, or the regular grid as a numpy array if `step` is False.
    """
    
    x = np.array([...])             # your 1D array of positions
    dx = np.diff(x)                 # differences between successive points
    #print(dx)                       # see all spacings
    #print("Median step:", np.median(dx))
    #print("Unique steps:", np.unique(np.round(dx, precision)))
    if average:
        dx = np.mean(dx)
    else:
        dx = np.median(dx)
    
    if step:
        return dx
    else:
        return np.arange(x.min(), x.max() + dx/2, dx)

    
def split_on_indices(vector,indices,absolute=False):
    """Split an array on a list of lengths (indices represent the number of items in each block). 
    Return a list of splitted sub-arrays.
    N.B.: the preferred implementation is block length and not absolute indices to avoid ambiguity in meaning of
    indices and first/last blocks. If absolute is set, indices are assumed as starting indices, last block is
    automatically included, if first one is wanted, must be explicetely included passing 0 as first index."""
    
    if absolute:
        istartblock = [b for a,b in zip([0]+indices[:-1],indices) if b-a!=1]+[len(vector)]
        indices = [b-a for a,b in zip([0]+istartblock[:-1],istartblock)]
    
    if np.sum(indices) != len(vector):
        warnings.warn ('len of indices for splitting is not matching vector length!')
    ind = np.cumsum(indices).tolist()
    result = [vector[i1:i2] for i1,i2 in zip([0]+ind,ind)]

    return result

def compare_len(testval):
    """Compare different methods related to  length of objects, useful for functions like `real_len`."""

    testval = [ 1,
    None,
    [2],
    [None],
    [1,2],
    [[1]],
    [[1,2],[3,4]],
    'aa',
    ['aa'],
    [1,2],
    [[1,2]],       
    ['abc','cde'],  
    [['abc','cde']],
    [],
    np.array([[1,2]]),
    np.array([1,2]),
    [np.array([1,2]),np.array([3,4])]    
    ]

    for t in testval:
        try:
            s = np.size(t)
        except TypeError:
            s = '-'
            
        try:
            sh = np.shape(t)
        except TypeError:
            sh = '-'
        
        try:
            l = len(t)
        except TypeError:
            l = '-'
        
        print(t,'|',s,sh,len(sh),l)
        

def real_len(x):
    """Return the real length of an object, intended as the outer level number of elements (strings counts as single elements).
    
    assert real_len([1,2])  == 2
    assert real_len([[1,2]] )== 1
    assert real_len(1)    == 1
    assert real_len([1])    == 1
    assert real_len(None)  ==  1
    assert real_len([None]) == 1
    assert real_len('abc')           == 1
    assert real_len(['abc'])          == 1
    assert real_len(['abc','cde'])          == 2
    assert real_len([['abc','cde']])          == 1
    
    assert real_len([1,2,3]) == 3
    assert real_len([]) == 1
    
    assert real_len([[1,2],[3,4]]) == 2
    
    assert real_len(np.array([[1,2]]))== 1
    assert real_len(np.array([1,2]))== 2
    assert real_len([np.array([1,2]),np.array([3,4])])== 2
    """
    
    if np.size(x)<=1:
        return 1
    
    return np.shape(x)[0]

    
def test_real_len():
    
    assert real_len([1,2])  == 2
    assert real_len([[1,2]] )== 1
    assert real_len(1)    == 1
    assert real_len([1])    == 1
    assert real_len(None)  ==  1
    assert real_len([None]) == 1
    assert real_len('abc')           == 1
    assert real_len(['abc'])          == 1
    assert real_len(['abc','cde'])          == 2
    assert real_len([['abc','cde']])          == 1
    
    assert real_len([1,2,3]) == 3
    assert real_len([]) == 1
    
    assert real_len([[1,2],[3,4]]) == 2
    
    assert real_len(np.array([[1,2]]))== 1
    assert real_len(np.array([1,2]))== 2
    assert real_len([np.array([1,2]),np.array([3,4])])== 2
    
    print('all tests passed!')


def vectorize(arg, n, force_if_n=False):
    """Vectorize a single array `arg` to length `n`. Typically used for vectorization of arguments (e.g. `Dlist`).
    
    An example of usage is when 
    
    N.B.: There are two ambiguities which need to be solved:
        1. n == len(arg)  (note that np.size is used instead of len to handle strings more correctly)
            In this case it is not possible to understand if the arg provided is single or multiple value,
            as this can be solved only understanding the expected input shape. 
            e.g.: [1,2] for a two-element input, is 1 for the 1st and 2 for the 2nd, or needs to be replicated? 
            By default `arg` is not replicated, the flag `force_if_n` overrides this behavior.
            2. same problem in a more subtle way when n = 1 and arg is scalar, in this case you probably want to convert the scalar in a single-element list, to iterate over it. If a different need is ever found a flag will be added
    
    examples (see `test_vectorize`):
        print(vectorize([1,2],3))  # [[1, 2], [1, 2], [1, 2]]
        print(vectorize(1,3))      # [1, 1, 1]
        print(vectorize(None,3))   # [None, None, None]
        print(vectorize([None],3)) # [[None], [None], [None]]
        print(vectorize([1],3))    # [[1], [1], [1]]
        print(vectorize([[1,2]],3))  # [[[1, 2]], [[1, 2]], [[1, 2]]]
        print(vectorize([1,2],2))   # [1, 2]
        print(vectorize([1,2],2,True)) # [[1, 2], [1, 2]]
        print(vectorize('abc',3))  # ['abc', 'abc', 'abc']
        print(vectorize(['abc'],3)) #[['abc'], ['abc'], ['abc']]

        print(vectorize(1,1))           # [1]
        print(vectorize(1,1,True))      # [1]
        print(vectorize([1],1))         # [1]
        print(vectorize([1],1,True))    # [[1]]
        
    to vectorize function arguments:
    
        args = [vectorize(a) for a in args]
    
    Note that force_if_n can be itself vectorized, e.g if this is passed together with a list of parameters and different behavior 
        also want to be passed and used consistently.
    
        args = [vectorize(a,f) for a,f in zip(args,vectorize(force_flags))]
    
    """
    
# <<<<<<< HEAD
    if (real_len(arg) != n) or force_if_n:        
        try:
            arg=[arg.deepcopy() for _ in range(n)]
        except AttributeError:  #fails e.g. if tuple which don't have copy method
            arg=[arg for _ in range(n)]  
# =======
#     #FIXME: this is still not a reliable mechanism, see extended tests below.
#     # in other cases used len(np.shape), maybe works better, e.g.:
#     # if len(np.shape(reader)) == 0: #vectorize #np.size(reader) == 1:
#     # reader=[reader]*len(rfiles)
    
#     if args:
#         for a in args:
#             if (np.size(a) == 1):
#                 args=[[] for i in range(n)]    
#             elif (len(a) != n):
#                 args=[args for i in range(n)]  
# >>>>>>> 9984c47
    else:
        if (n == 1) and len(np.shape(arg)) == 0:
            arg = [arg]    
    return arg


def test_vectorize():
    
    assert vectorize([1,2],3)  ==[[1, 2], [1, 2], [1, 2]]
    assert vectorize(1,3)    ==[1, 1, 1]
    assert vectorize(None,3)  ==[None, None, None]
    assert vectorize([None],3) ==[[None], [None], [None]]
    assert vectorize([1],3)    ==[[1], [1], [1]]
    assert vectorize([[1,2]],3)== [[[1, 2]], [[1, 2]], [[1, 2]]]
    assert vectorize([1,2],2)          == [1, 2]
    assert vectorize([1,2],2,True)          == [[1, 2], [1, 2]]
    assert vectorize('abc',3)           ==['abc', 'abc', 'abc']
    assert vectorize(['abc'],3)          ==[['abc'], ['abc'], ['abc']]
    assert vectorize(1,1)         ==   [1]
    assert vectorize(1,1,True)           ==  [1]
    assert vectorize([1],1)              ==   [1]
    assert vectorize([1],1,True)           ==  [[1]]
    
    assert vectorize([1,2,3], 2) == [[1,2,3],[1,2,3]]
    assert vectorize([], 3) == [[],[],[]]

    # <<<<<<< HEAD
    assert vectorize([1,2,3], 3, True) == [[1,2,3],[1,2,3],[1,2,3]]
    
    assert vectorize([[1,2],[3,4]],2) == [[1,2],[3,4]]
    assert vectorize([[1,2],[3,4]],2,True) == [[[1,2],[3,4]],[[1,2],[3,4]]]
    
    print('All tests passed.')
    # =======
    assert vectorize([1,2,3], 3) == [[1,2,3],[1,2,3],[1,2,3]]
    
    #extended test
    a = []
    b = 5
    c = [3]
    
    print("(a,3) --> ",vectorize(a,3))
    print("(a,1) --> ",vectorize(a,1))
    print("(b,3) --> ",vectorize(b,3))
    print("(b,1) --> ",vectorize(b,1))
    print("(c,3) --> ",vectorize(c,3))
    print("(c,1) --> ",vectorize(c,1))

    b = 'cane'
    c = ['cane']
    
    print("(b,3) --> ",vectorize(b,3))
    print("(b,1) --> ",vectorize(b,1))
    print("(c,3) --> ",vectorize(c,3))
    print("(c,1) --> ",vectorize(c,1))
    # >>>>>>> 9984c47


def inspect_vector(obj):
    """Print a set of dimensional information about the object.
    
    The returned information should be useful to discriminate between different lengths, dimensions and rank.
    
    """

    print("len: ",len(obj))
    print("np.size: ",np.size(obj))
    print("np.shape: ",np.shape(obj))
    print("len of shape: ",len(np.shape(obj)))

def make_raster(*vectors, as_axis = False, extend = True):
    """Create a raster grid on the base of a list of vectors.
    
    Vectors can be passed as axis if as_axis is False, in that case they are used as they are
        or as ranges, in which case they must be 3-vectors with start, stop, range, to be used as
        arguments for np.arange.
        In the second case, extremes can be included by extending the range by one additional step.
    
    TODO: vectorize as_axis and extend.
    """
    
    
    if extend and not as_axis:   vectors = [ (v[0], v[1] + v[2], v[2]) for v in vectors]
    gridaxis = (vectors if as_axis else [np.arange(*v) for v in vectors])    
    return list(product(*gridaxis))


def split_blocks (vector, sep = None, direction = None, split = False):

    """ Given a vector return the lenght of blocks (or blocks if split is True).
    
    A block is a monotonic sequence.
    
    The direction of monotony can be explicitly set with ``direction`` argument 
    (positive if increasing, negative if decreasing, 0 consider blocks as formed by equal elements).
    If direction is not provided, it is inferred from the first two elements.
    
    Return a list: if ``split`` is False (default), a list of the lengths of blocks is returned, otherwise a list of blocks is returned. 
    
    (e.g. [1,2,3,1,2,3] -> return [3,3])
    (e.g. [1,1,1,2,2,2,3,3,3,4,4,4], direction = 0 -> return [3,3,3,3]) 
    (e.g. [5,2,8,7,6,8], direction = 1 -> return [1,2,1,1,1])
    (e.g. [5,2,8,7,6,8], direction = None -> return [2,3,1])
    (e.g. [5,2,8,7,6,8], direction = -1 -> return [2,3,1]) # inferred from first two
    
    """
    
    if direction is None:  # exclude invalid points
        x = [vector[i] for i in np.where(np.isfinite(vector))[0]]
        direction = np.sign(x[1]-x[0])
    diff = np.diff(vector)
    
    if direction == 0:
        diff = np.insert(diff,0,0)
        ind = np.where(diff != 0)[0].tolist()
    else:
        direction = np.sign(direction)
        ind = list(np.where(np.sign(diff) != np.sign(direction))[0]+1)
    ind.append (len(vector))
    
    # correct for nan or invalid data, which artificially create two artificial changes of direction.
    # keep only the first, invalid data will be included in output in second block.
    ind = [i for i in ind if i in np.where(np.isfinite(vector))[0]+1]  # +1 to include nan in following block and keep into account first element if invalid. I suspect it will give problems if last element is invalid,
    #must be kept specifically into account.
    #if not np.isfinite(vector[0]): ind.remove(1)  # 1 is legitimate only if first two elements are finite
    
    #if not cumulative or split: # cumulative can be obtained with np.cumsum()
    ind = [i1-i2 for i1,i2 in zip(ind,[0]+ind[:-1])]
    
    
    
    return split_on_indices(vector, ind) if split else ind


def test_len_blocks():

    print(split_blocks([1,2,3,1,2,3]))   #[3,3]
    print(split_blocks([1,2,3,1,2]))   #[3,2]
    print(split_blocks([1,2,3,1,2],direction = -1))   #[1,1,2,1]
    print(split_blocks([1,2,3,1,2],direction = 0))   #[1,1,1,1,1]
    print(split_blocks([1,1,1,2,2,3,3,3],direction = 0))   #[3,2,3]
    
    print(split_blocks([1,2,3,1,2,3],split = True))   #[[1,2,3],[1,2,3]]
    print(split_blocks([1,2,3,1,2],split = True))   # [[1, 2, 3], [1, 2]]
    print(split_blocks([1,2,3,1,2],direction = -1,split = True))   # [[1], [2], [3, 1], [2]]
    print(split_blocks([1,2,3,1,2],direction = 0,split = True))   # [[1], [2], [3], [1], [2]]
    print(split_blocks([1,1,1,2,2,3,3,3],direction = 0,split = True))   # [[1, 1, 1], [2, 2], [3, 3, 3]]



def test_stats(*args,**kwargs):
    # generate random distribution
    n  = 1000
    
    v = np.random.random(n)
    
    print('len(data): ',len(v))   
    print("no options: all stats as float:\n")
    print(stats(v),"\n----------\n\n")    
    print("with empty vars, return empty:", stats(v,vars=[]),"\n----------\n\n")
    print("select two stats (vars=[0,1])", stats(v,vars=[0,1]),"\n----------\n\n")
    
    print("return as string (string=True), include units:\n", stats(v,string=True,vars = [2,6], units ='km'),"\n----------\n\n")    
    print("return as string (single string), that is appended to format according to default meaning:", stats(v,string=True,vars = [2,6],fmt = ['rms: %.6g A',
    'size: %.3i pts'],units='[A]'),"\n----------\n\n")  
    print("for finer control (e.g. more that one unit), it needs to be included in fmt)", stats(v,string=True,vars = [2,6],fmt = ['rms: %.6g A',
    'N: %.3i points']),"\n----------\n\n")
    
    print("if called without data, return a format string (units can be added, fmt is irrelevant)", stats(string=True,fmt=None,units='um',vars = [2,6]),"\n----------\n\n") 
    
    print("without data and vars return full format string", stats(string=True),"\n----------\n\n") 
 
    return stats(v,*args,**kwargs)
    
def is_nested_list(l):
    """Return true if it is a nested list (at least one element is a list), False otherwise (all elements are scalar).
    Uses isinstance, so arrays are not considered lists. 
    from https://stackoverflow.com/questions/24180879/python-check-if-a-list-is-nested-or-not."""
    #breakpoint()
    try:
          next(x for x in l if isinstance(x,list))
    
    except StopIteration:
        return False
    except TypeError: #l is scalar or None
        return False
    
    return True

def test_is_nested_list():
    """
    testvalues=[None,[None],[None,[],[]],[[1,2,3]],[3],[[3]],[[],[2],[3]],[[1],[2],[3]],[[1],[2],[3]],[[1],[2,3],[3]],[[2,3],[],[3]]]
    '''
    None | False
    [None] | False
    [None, [], []] | True
    [[1, 2, 3]] | True
    [3] | False
    [[3]] | True
    [[], [2], [3]] | True
    [[1], [2], [3]] | True
    [[1], [2], [3]] | True
    [[1], [2, 3], [3]] | True
    [[2, 3], [], [3]] | True
    '''
    """
    testvalues=[None,[None],3,[None,[],[]],[[1,2,3]],[3],[[3]],
        [[],[2],[3]],[[1],[2],[3]],[[1],[2],[3]],
        [[1],[2,3],[3]],[[2,3],[],[3]]]
    for tv in testvalues:
        #print('test string output:')
        print (tv,"|",is_nested_list(tv))

def is_iterable(obj):
    """ Check if an object is iterable.
    
    Apparently there is no standard way of doing it in Python, so this function is created to introduce
    a consistent check. Note that this is the suggested way, as checkng an __iter__ method is not enough
    because it can be iterated by __get_items__"""
    
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def test_make_raster(vectors):
    print (make_raster((1.,2,0.5),(3,4.,0.5),(2,4,1),extend = False)) 
    #[(1.0, 3.0, 2), (1.0, 3.0, 3), (1.0, 3.5, 2), (1.0, 3.5, 3), (1.5, 3.0, 2), (1.5, 3.0, 3), (1.5, 3.5, 2), (1.5, 3.5, 3)]
    print (make_raster((1.,2,0.5),(3,4.,0.5)))
    # [(1.0, 3.0), (1.0, 3.5), (1.0, 4.0), (1.5, 3.0), (1.5, 3.5), (1.5, 4.0), (2.0, 3.0), (2.0, 3.5), (2.0, 4.0)]    
    print (make_raster((1.,2,0.5),(3,4.,0.5), as_axis=True))
    # [(1.0, 3), (1.0, 4.0), (1.0, 0.5), (2, 3), (2, 4.0), (2, 0.5), (0.5, 3), (0.5, 4.0), (0.5, 0.5)]
    print (make_raster((1.,2,0.5),(3,4.,0.5), as_axis=True, extend = True))
    # [(1.0, 3), (1.0, 4.0), (1.0, 0.5), (2, 3), (2, 4.0), (2, 0.5), (0.5, 3), (0.5, 4.0), (0.5, 0.5)]


    
if __name__ == "__main__":
    test_stats()