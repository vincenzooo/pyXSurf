"""
2020/09/21
Differently from what done in pySurf, here we keep reader and register mechanism separated and leave to the user the responsibility to call them. They can be put together in future if a convenient model for readers is found.

`pyProfile.profile` has only a simple reader (a thin wrapper around `np.genfromtxt`) and a function `register_profile` to align and rescale x and y. Additional readers (`read_mx_profiles`, `read_xyz`) are considered experimental and kept here, even if they are functions.

2020/06/25 start to create profile class on template of data2D_class.
N.B.: there is an important difference between profile and data2D functions
work, and is that data2D functions accept as argument surfaces in the form (tuple)
`(data,x,y)`, while functions in `profile` want x and y as separate arguments.
Meaning that in this second case, passing only one argument is check with None
as default val on second argument, rather than with inspecting the elements together.
"""


"""
notes from data2D_class.py:
2018/06/07 v1.3
v1.2 was not convenient, switch back to same interface as 1.1:
don't modify self, return copy.

2018/06/06 v 1.2
After attempt to modify interface in a way that modify self, switch back to same interface as 1.1. It is inconvenient to have always to create a copy, in particular in interactive mode when same operation can be repeated multiple times you need to inizialize data every time (e.g. rotate).

2018/06/06 v1.1
methods are written to consistently return a copy without modifying self.
But this doesn't work, always need to assign, cannot chain (with property assignment, it works on methods) or apply to set of data in list.

Give some attention to inplace operators that can link to external data. At the moment when class is created from data, x, y these are assigned directly to the property resulting in a link to the original data. Some methods have inplace operations that will reflect on initial data, others don't."""

"""
programming notes:
-inplace methods-
A method acting on Data2D (e.g. level) can modify (o reassign to) self and return self or return a copy.

e.g. a method crop (or remove nancols) that return slices of data and x/y reassign to self.data/x/y views of original data.

A method that reassigns self.data=...
most of the time stays linked to original data.
If inplace changes (e.g. element-wise assignment) are then performed on the property, the change is reflected in the original data.
See notebook Python Programming Notes.

To remove this behavior, a copy of the data or view must be performed at some point (e.g. explicitly self.data.copy() or implicitly deepcopy(self).



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



import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
from copy import deepcopy
import inspect  # to use with importing docstring
import time

import dataIO
from dataIO.span import span
from dataIO.superlist import Superlist
from dataIO.arrays import split_blocks, split_on_indices
from dataIO.functions import update_docstring
from dataIO.fn_add_subfix import fn_add_subfix

from pyProfile.profile import crop_profile
from pyProfile.profile import level_profile
from pyProfile.profile import resample_profile
from pyProfile.profile import sort_profile
from pyProfile.profile import sum_profiles, subtract_profiles, multiply_profiles
from pyProfile.psd import psd as profpsd
from pyProfile.profile import movingaverage, rebin_profile
from pyProfile.profile import merge_profile
from pyProfile.profile import register_profile
from pyProfile.profile import save_profile

from pyProfile import profile   # functions passed to update_docstring will be from here
from pyProfile.psd import plot_psd
from pyProfile.psd import psd_units
from pyProfile.profile import fill_mask

'''
def update_docstring(func,source):
    """given a current function and a source function, update current function docstring
     appending signature and docstring of source.

     It provides a decorator @update_docstring(source) update the decorated
       function docstring. User should take care to create docstring of decorated function
       only as preamble to source docstrings, e.g.:
       this is func function, derived from source function by modifying the usage of parameter foo.
       parameter foo is changed of sign."""
    doc0="" if func.__doc__ is None else func.__doc__
    func.__doc__='\n'.join([doc0,source.__name__+str(inspect.signature(source)),source.__doc__])
    return func

def doc_from(source):
    """intent is to use partial to obtain a decorator from update_docstring.
    not sure it is the right way."""
    partial(update_docstring,func=func)
'''



def read_chr(fn):
    """ Read data from a crocodile file.
    
    Accept a string as filename, read 2nd and 3rd columns, replace ',' with '.' and convert to float.
    """

    def comma_to_float(s):
        return float(s.replace(',', '.'))

    data = np.genfromtxt(
        fn,
        delimiter='\t',
        skip_header=1,
        usecols=(2, 3),
        converters={2: comma_to_float, 3: comma_to_float},
        dtype=float,
        encoding='latin1',unpack=True
    )

    return data

def read_xyz(filename,*args,**kwargs):
    """tentative routine to read xyz files.
    Use by reading the data and then passing them to profile.
    It will be incorporated in some form of reader in a 
    more mature version"""

    raise NotImplementedError
    
def read_mx_profiles(filename,*args,**kwargs):
    """tentative routine to read xyz files. Return a list of all profiles
        with names from 3rd column (files are in format x,y,profilename).
    Use by reading the data and then passing them to profile.
    It will be incorporated in some form of reader in a 
    more mature version"""

    x,y,tag = np.genfromtxt(filename,delimiter=',',
                            unpack=True,skip_header=1,dtype=str,*args,**kwargs)
    x=x.astype(np.float)
    y=y.astype(np.float)
    labels = np.unique(tag)  # names of profiles in order
    groups = [np.where(tag==l)[0] for l in labels] # indices of points for each profile 
    profiles = [Profile(x[i],y[i],name=tag[i[0]]) for i in groups]
    
    return profiles


def get_header_field(db, tag, header='<<PMCA SPECTRUM>>'):
    """Extract a single field named `tag` as a list.
    tag must should match a field (column) name as inferred from header 
    (otherwise empty list is returned).
    Temporarily here for use with `read_mca`, can be moved to a more general
    use in dataIO.
    """
    
    from dataIO.read_pars_from_namelist import namelist_from_string
    headers = [d.header[header] for d in db['data']]
    return [namelist_from_string(t,separator='-')[tag] for t in headers]
    
def read_mca(filename, encoding='iso-8859-1',*args,**kwargs):
    """ Given a .mca file, return a `Profile` object.
    
    tentative routine to read mca files from amptek energy sensitive detector. Return a profile with metadata information in a `.header` property (temporarily a dictionary obtained from string blocks) of all profiles.
    Like all other readers will be incorporated in some form of reader in a more mature version.
    Normally the header contains keys
    ['<<CALIBRATION>>', 
    '<<PMCA SPECTRUM>>', 
    '<<DATA>>', 
    '<<DP5 CONFIGURATION>>', 
    '<<DPP STATUS>>']
    with the first one set to default in code (if data are uncalibrated, the key is missing).
    
    In the future can be included in a inherited Profile class with proper methods.
    
    """

    import re
    from scipy import interpolate

    with open(filename, 'r', encoding = encoding) as f:  # close is not needed
        content = f.readlines()   
    content = [l.strip() for l in content] # trimmed list of lines
    a = [aa for aa in content if len(aa)>0]   # list of not blank lines

    #  p=re.compile("<<.*>>")  # mi sembrano inutili
    #  i = p.match("".join(a))

    itags = [i for i,l in enumerate(a) if re.compile("<<.*>>").match(l)] #posizione dei tags in linee
    tags = [a[i] for i in itags]  # corresponding tags
    
    blocks = {'<<CALIBRATION>>':['LABEL - Channel','0 0.','1 1']} #default calibration if not defined in file, in a consistent format for conversion.
    for i,t in enumerate(tags[:-1]):  #last tag is assumed to be closing tag
        if itags[i+1] != itags[i]+1:
            blocks [t] = a[itags[i]+1:itags[i+1]]  
    #pdb.set_trace()
    #print(blocks['<<CALIBRATION>>'])
    data = [float(d) for d in blocks['<<DATA>>']  ]
    cal = np.array([[float(dd) for dd in d.split()] for d in blocks['<<CALIBRATION>>'][1:]])
    #print(len(data),cal)
    x = interpolate.interp1d(cal[:,0],cal[:,1],fill_value='extrapolate')(np.arange(len(data))) #use interp1d because np.interp doesn't extrapolate
    #print(span(x),len(x))
    
    #print(cal)
    profiles = Profile(x,data,name=filename)
    profiles.header = blocks
    
    return profiles


class Profile(object):  #np.ndarrays
    """A class containing x,y data. It has a set of methods for analysis and visualization.
    Function methods: return a copy with new values.
    to use as modifier (procedure), e.g.:
        a=a.level() """

    """C'era qui un tentativo fallito di subclassare nd.array
    usando __new__ e __array_finalize__."""


    def __init__(self,x=None,y=None,file=None,reader=None,
        scale = (1.,1.), units=None,name=None,*args,**kwargs):
        """can be initialized with data; x,y; file; file, x
        if x is provided, they override x from data if matching number of elements, 
        or used as range if two element (error is raised in case of ambiguity)."""
        
        #from pySurf.readers.instrumentReader import reader_dic
        from pyProfile.profile import register_profile
        
        #pdb.set_trace()

        if isinstance (x,str):
            print ('first argument is string, use it as filename')
            file=x
            x=None
        else:
            if y is not None:
                y=np.array(y) #convert to array if not
                x=np.array(x)
        # import pdb
        # pdb.set_trace()
        self.file=file #initialized to None if not provided
        if file is not None:
            assert y is None
            # passed file AND xrange, overrides x because only range matters.
            #store in xrange values for x if were passed
            xrange=span(x) if x is not None else None
            #pdb.set_trace()
            
            if reader is not None: raise NotImplementedError("readers are not implemented yet for profiles,"+
                "\tPass data or read from two column text file in compatible format.")
            
            #pdb.set_trace()
            self.load(file,*args,**kwargs)
            """
            if reader is None:
                reader=auto_reader(file) #returns a reader
            """
            from pyProfile.profile import load_profile
            x,y=load_profile(file,*args,**kwargs) #calling without arguments skips register, however skips also reader arguments, temporarily arranged with pop in read_data to strip all arguments for
                #register data and pass the rest to reader
            #pdb.set_trace()
            #TODO: Cleanup this part and compare with pySurf. Consider centering
            # of axis coordinates as in grid example.
            if np.size(xrange) == np.size(y):
                # x is compatible with data size
                if np.size(xrange) == 2:
                    print ('WARNING: You passed both data and x, with size 2. There may be some ambiguity in how coordinates are handled')
                self.x=xrange
            elif np.size(xrange) == 2:
                #here it must be intended as a range. Should check also interval centerings.
                #it was considered somewhere in pySurf.
                x=np.linspace(*xrange,np.size(y))
            elif xrange is not None:
                print("wrong number of elements for x (must be 2 or xsize_data), it is instead",np.size(xrange))
                raise ValueError
            
            # set self.header to file header if implemented in reader, otherwise set to empty string""
            #questo dovrebbe sempre fallire
            try:
                #kwargs['header']=True
                self.header=reader(file,header=True,*args,**kwargs)
            except TypeError:  #unexpected keyword if header is not implemented
                self.header=""
                #raise
        else:
            if y is not None:
                if len(y.shape) != 1:
                    #pdb.set_trace()
                    print('WARNING: data are not unidimensional, results can be unpredictable!')
                if x is None:
                    x=np.arange(np.size(y))  

    #if data is not None:
        # self.x, self.y = np.array(x), np.array(y)
        
        # breakpoint()
        self.x,self.y=register_profile(np.array(x), np.array(y),scale=scale,*args,**kwargs) # se load_profile calls register, this
        #goes indented.

        if np.size(units) == 1:
            if units is not None:
                units=[units,units]
        self.units=units
        if name is not None:
            self.name=name
        elif file is not None:
            self.name=os.path.basename(file)
        else:
            self.name=""
        #print(name)

    def __call__(self):
        return self.x,self.y
    """
    def __add__(self,other,*args,**kwargs):
        return Profile(*sum_profiles(*self(),*other(),*args,**kwargs),units=self.units,name=self.name + " + " + other.name)
    """

    def __add__(self,other,*args,**kwargs):
        # print('self (id,type):',id(type(self)),type(self))
        # print('other (id,type):',id(type(other)),type(other))
        # print('Profile (id,type):',id(Profile),type(Profile))
        # print(isinstance(other,Profile))
        # print(other is Profile)
        
        if isinstance(other,self.__class__):
            res = sum_profiles(*self(),*other(),*args,**kwargs)
            #res = other()
            res = self.__class__(*res,units=self.units,name=self.name + " + " + other.name)
        else:
            try:
                res = self.copy()
                res.y = res.y + other
                res.units = self.units
            except ValueError:
                raise ValueError("Unrecognized type in sum")
        return res
        
    def __sub__(self,other,*args,**kwargs):
        """
        This is made on copy and paster from add. It is maybe not the most flexible way in relation with multiplication and operations with different types, but it works well enough.
        
        original code (handle only profile objects):       
            res=Profile(*(subtract_profile(*self(),*other(),*args,**kwargs))(),units=self.units)
            res.name = self.name + " - " + other.name
            return res
        """
        #  y = np.genfromtxt(fn)
        #p=Profile(np.arange(len(y)),y)
        #p.plot(label = 'raw')
        #
        #p.level().plot('r',label = 'leveled')
        #(p-p.level()).plot('r',label = 'fit')
        #plt.legend()
        #
        #pdb> isinstance(other,Profile)
        #   False

        #ipdb> other
        #   <pyProfile.profile_class.Profile object at 0x000002A55C104CF8>

        #ipdb> Profile
        #<class 'pyProfile.profile_class.Profile'>
        """
        print ("class: ",other.__class__)
        # Out[65]: pyProfile.profile_class.Profile

        print("check: ",other.__class__ == Profile)
        # Out[66]: True

        print ("type: ",type(other))
        # Out[65]: pyProfile.profile_class.Profile

        print("check: ",type(other) == Profile)
        #Out[67]: True

        print("check: ", isinstance(other,Profile))
        #Out[68]: True
        """
        
        #pdb.set_trace()
        
        if isinstance(other,self):
            res = subtract_profiles(*self(),*other(),*args,**kwargs)
            res = self.__class__(*res,units=self.units,name=self.name + " - " + other.name)
        else:
            try:
                res = self.copy()
                res.y = res.y - other
                res.units = self.units
            except ValueError:
                raise ValueError("Unrecognized type in subtraction")
        return res
    
    def __repr__(self):
    
        return '<.Profile "%s" at %s>'%(self.name,hex( id(self)))
        
        ''' N.B.: you cannot retrieve and modify existing __repr__,
        the following gives infinite recursion, the only way is to recreate a new repr: 

            class me():
                def __repr__(self):
                    return self.__repr__()
                    
            a = me()
            a'''
    
    # def __add__(self,other,*args,**kwargs):
    #     print(type(other))
    #     print(Profile)
    #     print(isinstance(other,Profile))
    #     print(type(Profile))
    #     if isinstance(other,Profile):
    #         res = sum_profiles(*self(),*other(),*args,**kwargs)
    #         res = Profile(*res,units=self.units,name=self.name + " + " + other.name)
    #     else:
    #         try:
    #             res = self.copy()
    #             res.y = res.y + other
    #             res.units = self.units
    #         except ValueError:
    #             raise ValueError("Unrecognized type in sum")
    #     return res
    
    def __mul__(self,scale,*args,**kwargs):
        """mutiplocation: accept Profile, scalars or 2-vector [x,y]. """
        
        # this is called directly if self is a Profile object.
        # mul gets called if self and scale are same type,
        # or if self can handle self.__mul__(scale)
        # when this doesn't work, it calls scale.__rmul__(self)
        # so for example:
        # p1*p2 calls p1.mul(p2) -> ok
        # p2*5 calls p2.mul(5) -> ok
        # 5*p2 calls p2.__rmul__(5) <-- fallback from 5.__mul__(p2)  
        
        res = self.copy()
        
        
        #breakpoint()
        if isinstance(scale,self.__class__):
            #if np.size(scale)==1:  # scalar object or value
            
            """if it is Profile, do pointwise multiplication rescaling on firts."""
            # resample and multiply. For surface result, use matrix multiplication.
            #raise NotImplementedError ('can be ambigous (return point to point multipl. or surface? Fix in code, at the momeb accept only x and (y) scalars.')
            
            res.x,res.y = multiply_profiles(*self(),*scale(),*args,**kwargs)
            #res = Profile(*res,units=self.units,name=self.name + " + " + scale.name)
            #tmp = scale.resample(self,trim=False,left = np.nan, right = np.nan) # this can have fewer points  
            #res=self.copy()
            #res.y = self.y * tmp.y
            
            #FIXME questo non convince, ad es, se none, ad es se divisione
            # changed 2022/11/30 to implement None
            # see also how it is handled if units is None
            u1 = '' if res.units[1] is None else res.units[1]
            u2 = '' if scale.units[1] is None else scale.units[1]
            if u1 == u2:
                res.units[1] = u2+'^2'
            else:
                res.units[1]='%s %s'%(u1,u2)
                
            if self.name and scale.name:
                res.name = self.name + ' x ' + scale.name
            else: 
                res.name = (self.name if self.name else scale.name) + ' product'
        elif np.size(scale)==2:      # x and y scales      
            res.x = scale[0] * res.x
            res.y = scale[1] * res.y
        else:     # era value if np.size(scale) != 1
            #breakpoint()
            res.y = scale * res.y 
            if self.name:
                res.name = '%s x %s'%(self.name,scale)
            else:
                res.name = 'x %s'%scale
        # else:
        #     raise ValueError('Multiply Profile by wrong format!')
        return res
        '''
    
    
        #breakpoint()
        if np.size(scale)==1:  # scalar object or value
            # print(type(scale))
            # print(Profile)
            # print(isinstance(scale,Profile))
            # print(type(Profile))
            if isinstance(scale,self.__class__):
                """if it is Profile, do pointwise multiplication rescaling on firts."""
                # resample and multiply. For surface result, use matrix multiplication.
                #raise NotImplementedError ('can be ambigous (return point to point multipl. or surface? Fix in code, at the momeb accept only x and (y) scalars.')
                
                res = multiply_profiles(*self(),*scale(),*args,**kwargs)
                #res = Profile(*res,units=self.units,name=self.name + " + " + scale.name)
                #tmp = scale.resample(self,trim=False,left = np.nan, right = np.nan) # this can have fewer points  
                #res=self.copy()
                #res.y = self.y * tmp.y
                
                #FIXME questo non convince, ad es, se none, ad es se divisione
                # changed 2022/11/30 to implement None
                # see also how it is handled if units is None
                u1 = '' if res.units[1] is None else res.units[1]
                u2 = '' if scale.units[1] is None else scale.units[1]
                if u1 == u2:
                    res.units[1] = u2+'^2'
                else:
                    res.units[1]='%s %s'%(u1,u2)
                    
                if self.name and scale.name:
                    res.name = self.name + ' x ' + scale.name
                else: 
                    res.name = (self.name if self.name else scale.name) + ' product'
            else:     # value
                #breakpoint()
                res.y = scale * res.y 
                if self.name:
                    res.name = '%s x %s'%(self.name,scale)
                else:
                    res.name = 'x %s'%scale
        elif np.size(scale)==2:      # x and y scales      
            res.x = scale[0] * res.x
            res.y = scale[1] * res.y
        else:
            raise ValueError('Multiply Profile by wrong format!')
        return res
        '''
        
    def __rmul__(self,scale,*args,**kwargs):
        # this is called on the second term (self) if mul of the first (scale) failed in scale * self.
        
        #print('__rmul__ (%s,%s) calling %s__mul__ (%s)'%(2*(self,scale)))
        
        res = self.__mul__(scale,*args,**kwargs)
        #breakpoint()
        res.name = '%s x '%(scale)+(self.name if self.name else '') 
        return res
        
    def __matmul__(self,scale,*args,**kwargs):
        raise NotImplementedError ('Return a 2D surface (Data2D).')
        
    def __neg__(self):
        return self.__mul__(-1)

    def __rtruediv__(self,other):
        """division: accept Profile, scalars or 2-vector [x,y]. """
        # we are here because other/self didn't succeed with other.__truediv__
        # self is Profile, other can be integer or profile. 
        # it is result = other/self
        # print(self)
        # print(other)
        
        res = self.copy()
        if np.size(other) == 1:
            if isinstance(other,self.__class__):  # scalar
                sel = (self.y != 0) 
                res = self.copy()
                res.y[sel] = other*(1./self.y[sel])
                res.y[~sel] = np.nan
                #breakpoint()
                if other.units[1] == self.units[1]:
                    res.units[1] = ''  # adimensional
                else:
                    u1 = '' if res.units[1] is None else res.units[1]
                    u2 = '' if other.units[1] is None else other.units[1]
                    res.units[1]='%s / %s'%(u1,u2)
                res.name = '%s ratio'%(other.name if other.name is not None else '' ) + (' / %s'%(self.name) if self.name is not None else '') 
            else:     # value
                #breakpoint()
                res.y = other / res.y 
                
                if self.units[1]:
                    res.units[1] = '/'+self.units[1] 
                if self.name:
                    res.name = '%s / %s'%(other,self.name)
                else:
                    res.name = '%s / ratio'%(other)

        elif np.size(other)==2:      # x and y others    
            #return other.__truediv__(self)
            res.x = other[0] / res.x
            res.y = other[1] / res.y
        else:
            raise ValueError('__rtruediv__ Profile by wrong format!')
        return res

    def __truediv__(self,other):
    
        # __truediv__ is normally called, 
        # gets here for p2/p1 # p2/p1 = __truediv__(self=p2,other=p1)
        # gets here for p2/5 # p2/5 = __truediv__(self=p2,other=5)
        # 5/p2 will not call this, but will try to call 5.__truediv__, if it fails,
        #     p2.__rtruediv__ will be called instead

        '''https://stackoverflow.com/questions/37310077/python-rtruediv-does-not-work-as-i-expect
        
        __rtruediv__ only has priority over __truediv__ if the right-hand operand is an instance of a subclass of the left-hand operand's class.

        When you do 343 / x, NewInt is a subclass of int, so x's __rtruediv__ gets priority. When you do 343.3 / x, NewInt is not a subclass of float, so 343.3's __truediv__ gets priority.

        343.3.__truediv__(x) doesn't return NotImplemented, since float knows how to divide a float by an int. Thus, x.__rtruediv__ doesn't get called.
        '''
        # __rtruediv__ is implemented in Profile, so 1/other makes sense for Profile and numbers. 
        #breakpoint()
        res = self*(1./other)
        u = getattr(other,'units',['',''])[1]  # assegna units y or ''
        
        if self.units[1] == u:
            res.units[1] = ''   # self and other have same units, cancel out
        else:
            if self.units[1] is None:
                self.units[1] = ''
            res.units[1] = self.units[1]+('/'+u if u else '')
        if self.name is not None:
            #res.name = '__truediv__'
            res.name = '%s%s'%(self.name if self.name else "ratio",((' / %s'%getattr(other,'name',other)) if getattr(other,'name','') else ''))
        return res

    
    def merge(self,other,*args,**kwargs):
        
        if isinstance(other,self.__class__):
            #res = merge_profiles([[self.x,self.y],[other.x,other.y]],*args,**kwargs)
            res = merge_profile(self.x,self.y,other.x,other.y,*args,**kwargs)
            res = self.__class__(*res,units=self.units,name=self.name + " // " + other.name)
        else:
            raise ValueError("Unrecognized type in merge")
        return res
        
    def min (self):
        return np.nanmin(self.y)

    def max (self):
        return np.nanmax(self.y)    
        
    def sort (self,reverse=False):
        """return sorted copy."""
        res = self.copy()
        res.x,res.y= sort_profile(self.x,self.y,reverse=reverse)   
        return res
    
    def plot(self,title=None,*args,**kwargs):
        """plot profile using and setting automatically labels.
        
        Additional arguments are passed to `plt.plot`.
        Quite useless for profile, can be plot with `plt.plot(*P(),*args,**kwargs)"""
        
        #modeled over data2D.plot, non c'e' plot_profile
        from plotting.captions import legendbox
        from pyProfile.profile import get_stats
        
        stats=kwargs.pop('stats',0) #to change the default behavior
        loc=kwargs.pop('loc',0) #location for stats legend
        framealpha=kwargs.pop('framealpha',0.5) #transparency for stats legend
        
        # N.B. this must allow to pass None to omit plotting,
        #   and use name if label is undefined and name is defined.
        
        l=None
        try:
            l = kwargs.pop('label',self.name)
        except UnboundLocalError:
            l = None
        
           
        res=plt.plot(self.x,self.y,label=l,*args,**kwargs)
        
        if stats: #add stats to plot
            legend=get_stats(self.x,self.y,units=self.units)
            l=legendbox(legend,loc=loc,framealpha=framealpha)
            
        plt.xlabel('X'+(" ("+self.units[0]+")" if self.units[0] else ""))
        plt.ylabel('Y'+(" ("+self.units[1]+")" if self.units[1] else ""))
  
        if title is None:
            if self.name is not None:
                title = self.name
        plt.title(title)
        
        return res
        
    plot=update_docstring(plot,plt.plot)


    def load(self,filename,*args,**kwargs):
        """A simple file loader using np.genfromtxt.
        Load columns from file in self.x and self.y."""
        #pdb.set_trace()
        self.x,self.y = np.genfromtxt(filename,unpack=True,*args,**kwargs)
    load=update_docstring(load,np.genfromtxt)

    def save(self,filename,*args,**kwargs):
        """Save data using `pyProfile.profile.save_profile`."""
        
        # if not explicitly set, header is built joining units with the proper delimiter
        delimiter = kwargs.pop('delimiter','\t')
        h = kwargs.pop('header',delimiter.join(self.units) if self.units else None) 
        res = save_profile(filename,self.x,self.y,header = h,*args,**kwargs)
        return res
    save.__doc__=save_profile.__doc__
    

    def register(self,filename,*args,**kwargs):
        """Use pyProfile.profile.register_profile to rescale."""
        #pdb.set_trace()
        self.x,self.y = register_profile(x,y,*args,**kwargs)
    register=update_docstring(register,register_profile)

    '''
    from functools import update_wrapper
    #@update_wrapper(rotate_data)  #this doesn't work as I would like
    def rotate(self,angle,*args,**kwargs):
        """call data2D.rotate_data, which rotate array of an arbitrary angle in degrees in direction
        (from first to second axis)."""
        res = self.copy()
        res.data,res.x,res.y=rotate_data(self.data,self.x,self.y,angle,*args,**kwargs)
        return res
    rotate=update_docstring(rotate,rotate_data)
    '''

    '''
    def apply_to_data(self,func,*args,**kwargs):
        """apply a function from 2d array to 2d array to data."""
        res = self.copy()
        res.data=func(self.data,*args,**kwargs)
        return res
    '''
    
    def crop(self,*args,**kwargs):
        """crop profile making use of function profile.crop_data, where x,y are taken from self."""
        
        res=self.copy()
        res.x,res.y=crop_profile(self.x,self.y,*args,**kwargs)
        return res #
    crop=update_docstring(crop,crop_profile)
    
    def movingaverage(self,*args,**kwargs):
        """moving average using function profile.movingaverage, where x,y are taken from self."""
        
        res = self.copy()
        res.y = movingaverage(self.y,*args,**kwargs)
        return res #
    movingaverage = update_docstring(movingaverage,movingaverage)
        
    def rebin(self,*args,**kwargs):
        """rebin using function profile.rebin_profile, where x,y are taken from self."""
        
        res=self.copy()
        res.x,res.y=rebin_profile(self.x,self.y,*args,**kwargs)
        return res #
    
    rebin=update_docstring(rebin,rebin_profile)
    
    def level(self,degree=1,zero='mean',*args,**kwargs):
        """return a leveled profile calling profile.level_profile.
`zero option can be 'top', 'bottom' or 'mean', and is a facility to shift curves so that their min or max value is aligned to zero."""
        
        res=self.copy()
        res.x,res.y=level_profile(self.x,self.y,degree=degree,*args,**kwargs)
        if zero == 'top':
            res.y = res.y - np.nanmax(res.y)
        elif zero == 'bottom':
            res.y = res.y - np.nanmin(res.y)
        elif zero != 'mean':
            raise ValueError ("wrong value for `zero` option in level")
        return res
    level=update_docstring(level,level_profile)

    def resample(self,other,*args,**kwargs):
        """TODO, add option to pass x and y instead of other as an object."""
        res=self.copy()
        try:
            if self.units is not None and other.units is not None:
                if self.units[0] != other.units[0]:
                    raise ValueError('If units are defined they must match in Profile.resample.')
            res.x,res.y=resample_profile(*res(),*other(),*args,**kwargs)   
        except AttributeError: #assume other is an array
            res.x,res.y=resample_profile(*res(),other,*args,**kwargs) #try with other array of x points
        return res        
    resample=update_docstring(resample,resample_profile)


    def psd(self,wfun=None,rmsnorm=True,norm=1):
        """return a PSD object with psd of self. """

        f,p=profpsd(self.x,self.y,wfun=wfun,norm=norm,rmsnorm=rmsnorm)

        return PSD(f,p,units=self.units,name="")
    psd=update_docstring(psd,profpsd)

    def remove_nan_ends(self,*args,**kwargs):
        res = self.copy()
        res.x,res.y=profile.remove_nan_ends(self.x,self.y,*args,**kwargs)
        return res
    remove_nan_ends=update_docstring(remove_nan_ends,profile.remove_nan_ends)

    def std(self):
        """return standard deviation of data excluding nans.
        TODO: it should be weighted average over x interval for non-equally spaced points."""
        return np.nanstd(self.y)

    def copy(self):
        """copy.deepcopy should work well."""
        return deepcopy(self)

    def printstats(self,label=None,fmt='%3.2g'):
        if label is not None:
            print(label)
        s=("%s PV: "+fmt+", rms: "+fmt)%(self.name,span(self.y,size=True),
                             np.nanstd(self.y))
        print(s)
        return s
    
    from dataIO.outliers import remove_outliers as _remout
    def remove_outliers(self,fill_value=np.nan,correct=False,*args,**kwargs):
        """use dataIO.outliers.remove_outliers to remove outliers.
        If correct is True, replace outliers with interpolated values (fill_value is ignored)."""
        
        res=self.copy()
        m = dataIO.outliers.remove_outliers(res.y,*args,**kwargs)
        # import pdb
        # pdb.set_trace()
        
        if correct:
            res.y = fill_mask(res.y,res.x,mask=m,extrapolate= True)
        else:
            res.y[~m] = fill_value
        return res
        
    remove_outliers=update_docstring(remove_outliers,_remout)

    '''
    def align_interactive(self,other,find_transform=find_affine):
        """interactively set markers and align self to other.
        Alignment is performed using the transformation returned by
           find_transform(markers1,markers2) after markers are interactively set.
        Return aligned Data2D object.
        There is an experimental version for dlist in scripts."""
        from pySurf.scripts.dlist import add_markers
        m1,m2=add_markers([self,other])
        trans=find_transform(m1,m2)
        return self.apply_transform(trans)

    def histostats(self,*args,**kwargs):
        res =data_histostats(self.data,self.x,self.y,units=self.units,*args,**kwargs)
        plt.title(self.name)
        return res
    histostats=update_docstring(histostats,data_histostats)

    def slope(self,*args,**kwargs):
        #import pdb
        #pdb.set_trace()
        scale=kwargs.pop('scale',None)
        if self.units is not None:
            if scale is None:
                if self.units[0]==self.units[1]:  #check if x and y in mm and z in um.
                    if self.units[0]=='mm' :
                        if self.units[0]=='mm' and self.units[2]=='um': scale=(1.,1.,1000.)
                else:
                    raise ValueError("x and y different units in slope calculation")
        else:
            scale=(1.,1.,1.)


        say,sax=slope_2D(self.data,self.x,self.y,scale=scale,*args,**kwargs)

        return Data2D(*sax,units=[self.units[0],self.units[1],'arcsec'],name=self.name + ' xslope'),Data2D(*say,units=[self.units[0],self.units[1],'arcsec'],name=self.name + ' yslope')
    slope=update_docstring(slope,slope_2D)
    '''

#from pySurf.psd2d import psd2d,plot_psd2d


        
class PSD(Profile):
    """It is a type of profile with customized behavoiur and additional properties
    and methods."""
    def __init__(self,*args,**kwargs):
        ''' super is called implicitly (?non vero)
        """needs to be initialized same way as Data2D"""
        #if a surface or a wdata,x,y are passed, these are interpreted as
        super().__init__(*args,**kwargs)
        '''
        super().__init__(*args,**kwargs)
                
    def plot(self,*args,**kwargs):
        u=kwargs.pop('units',self.units)
        l=kwargs.pop('label',self.name)
        return plot_psd(self.x,self.y,units=u,label=l,*args,**kwargs)
    
    def rms_power(self,plot=False,*args,**kwargs):
        """Calculate rms slice power as integral of psd. If plot is set also plot the whole thing."""
        
        raise NotImplementedError
        
        '''
        if plot:
            return plot_rms_power(self.y,self.data.self.x,units=self.units,*args,**kwargs)
        else:
            """this is obtained originally by calling rms_power, however the function deals with only scalar inmot for rms range.
            Part dealing with multiple ranges is in plot_rms_power, but should be moved to rms_power."""
            raise NotImplementedError
        '''
        
    def save(self,filename,*args,**kwargs):
        """Save psd."""
        super().save(filename,header='# f[%s] PSD[%s]'%tuple(psd_units(self.units)))

def load_from_blocks(file, *args, **kwargs):
    """read a list of profiles from a single file, extracting them using ``split_blocks``."""
    
    x,y = np.genfromtxt(file, unpack = True, *args, **kwargs)
    i = split_blocks(x)
    xx = split_on_indices(x,i)
    yy = split_on_indices(y,i)
    profiles = [Profile(x, y, name = '%03i'%i) for i,(x,y) in enumerate(zip(xx,yy))]
    return profiles
    
def load_plist(rfiles,reader=None,*args,**kwargs):
    """Read a set of profile file(s) to a plist. By default, files are split on blocks with spaces or changes in monotony.
    
    readers and additional arguments can be passed as scalars or lists,
    including arguments for ``load_from_blocks``.
    Return a Plist.
    
    2022/06/28 first implementation, copying from scripts/dlist.py, docstring not yet updated
    Mechanism for args and kwargs is quite basic and error-prone.

    You can pass additional arguments to the reader in different ways:
     - pass them individually, they will be used for all readers
         load_dlist(.. ,option1='a',option2=1000)
     - to have individual reader parameters pass them as dictionaries (a same number as rfiles),
         load_dlist(.. ,{option1='a',option2=1000},{option1='b',option3='c'},..)
         in this case reader must be explicitly passed (None is acceptable value for auto).

    Example:
        plist=load_plist(rfiles,reader=fitsWFS_reader,scale=(-1,-1,1),
                units=['mm','mm','um'])

        dlist2=load_dlist(rfiles,fitsWFS_reader,[{'scale':(-1,-1,1),
                'units':['mm','mm','um']},{'scale':(1,1,-1),
                'units':['mm','mm','um']},{'scale':(-1,-1,1),
                'units':['mm','mm','$\mu$m']}])
    """
    
    if isinstance(rfiles, str):
        profiles = load_from_blocks(rfiles, *args, **kwargs)
    else:
        if reader is None:
            #reader=auto_reader(rfiles[0])
            #reader = [auto_reader(r) for r in rfiles]
            reader = [None for r in rfiles]  # placeholder
            
        if np.size(reader) ==1:
            reader=[reader]*len(rfiles)
            
        ''' additional options see Dlist '''
        
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
        # print(kwargs)
        if not args:  #assume is correct number of elements
            args = [[]]*len(rfiles)
        
        #pdb.set_trace()

        #transform vectorized kwargs in list of kwargs
        kwargs=[{k:v[i] for k,v in kwargs.items()} for i in np.arange(len(rfiles))]
        profiles = [Profile(file=wf1,reader=r,*a,**k) for 
            wf1,r,a,k in zip(rfiles,reader,args,kwargs)]
        
    self = Superlist()
    for p in profiles:    
        self.append(p)
        
    return self

'''
class Plist(Superlist):
    """A list of Profile objects on which unknown operations are performed serially.
    Useless, it is just a Superlist of profiles."""
    
    """
    def plot(self,*args,**kwargs):
        """plot over same plot each profile in the list."""
        
        for p in self:
            p.plot()
        
        return plt.gca()  
    """
    
    def __init__(self, files = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        if files is not None:
            self = load_plist(files, *args, **kwargs)
'''
        
from dataIO.superlist import Superlist
from dataIO.arrays import vectorize


class Plist(Superlist):

    def __init__(self, *args, **kwargs):
        
        if 'files' in kwargs:
            files = kwargs.pop('files')
            s = load_plist(files, *args, **kwargs)   # return a list
            super().__init__(s)
        else:
            super().__init__(*args, **kwargs)        
    
    # assign method from dlist, which is sintactically identical
    from pySurf.scripts.dlist import Dlist
    plot = Dlist.plot
    
    """ N.B.: TODO: could use linecollections for plotting https://matplotlib.org/stable/api/collections_api.html#matplotlib.collections.LineCollection
    """
    
    def merge(self, mode='raw',*args,**kwargs):
        
        mode = vectorize(mode, len(self)-1,force_if_n=True)
        res = self[0]
        for p,m in zip(self[1:],mode):
            res = res.merge(p,mode=m,*args,**kwargs)
        return res
    
    def save(self,file, mode= 'vstack', *args,**kwargs):
        """save stacked or on separate files"""
        
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

        if mode == 'separate':
            outfiles=[]
            for p in self:
                name=fn_add_subfix(file,p.name)
                outfiles.append(name)
                p.save(name, *args,**kwargs)
            return outfiles
        elif mode == 'blocks':   #vstack or merged
            pp = self.merge()
            print("blocks mode not implemented yet")
            raise NotImplementedError('blocks mode not implemented yet in Plist.save')
        elif mode == 'vstack':
            pp = self.merge()
            pp.save(file, *args,**kwargs)
        else:
            raise ValueError('Unknown mode %s in Plist.save'%mode)
        
        return file    
    
def test_load_plist(rfiles = None):

    if rfiles is None:
        rfiles =[r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706.txt',
         r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706_sm11.txt',
         r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706_sm31.txt']

    pl = load_plist(rfiles,reader=None,units=['mm','um'],delimiter=',')
    #a.plot()
    
    print('\ntest method which perform actions (plot):')
    print(pl.plot())
    print('\ntest method which returns values (profile.min):')
    print(pl.min())
    print('\ntest chained method which returns Superlist (profile.crop):')
    pl.crop([2,3]).plot()
    
    """
    plist2=load_plist([],None,[{'scale':(-1,-1,1),
            'units':['mm','um']},{'scale':(1,1,-1),
            'units':['mm','um']},{'scale':(-1,-1,1),
            'units':['mm','$\\mu$m']}])
    return plist,plist2
    """
    
    return pl

def test_plist():

    rfiles =[r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706.txt', r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706_sm11.txt', r'C:\Users\kovor\Documents\python\pyXTel\source\pyProfile\test\input_data\01_mandrel3_xscan_20140706_sm31.txt']
    
    p = Plist(files = rfiles, delimiter = ',')
    p.plot()
    plt.title('Plist initialized with files')

    plt.figure()
    p = Plist([ Profile(file = f, delimiter = ',') for f in rfiles])
    p.plot()
    plt.title('Initialized with Profiles read at init')

    plt.figure()
    profiles = load_plist(rfiles, delimiter = ',')
    p = Plist(profiles)
    p.plot()
    plt.title('Initialized with profiles loaded with function')
    
    # interessante questa e' plist
    a=p.max()
  
def test_class_init(wfile=None):
    """test init and plot"""
    
    x1,y1 = load_test_data(wfile,*args,**kwargs)

    plt.figure(1)
    plt.clf()
    plt.suptitle(relpath)
    plt.title('use plot_data function')
    plot_profile(x1,y1,aspect='equal')

    a=Profile(x1,y1)
    plt.figure(2)
    plt.clf()
    plt.title('From data')
    a.plot(aspect='equal')

    b=Profile(file=wfile,center=(0,0))
    plt.figure(3)
    plt.clf()
    plt.title('from filename')
    b.plot(aspect='equal')

    #b.save(os.path.join(outpath,os.path.basename(fn_add_subfix(relpath.as_posix(),"",".txt"))), makedirs=True)
    b.remove_nan_ends()
    plt.figure()
    plt.title('removed nans')
    b.plot(aspect='equal')
    
    
def test_profile_mul(p1,p2):
    #test mul
    print(p1)
    print(p1*p2) #calls p1.mul(p2) -> ok
    print(5*p1) #calls 5.mul(p2) -> falls back on p2.__rmul__
    print(p1*5) #calls p1.mul(5) -> ok
    
    #test mul
    print(span(p1.y))
    print('---------------')
    print(span((p1*p2).y)) #calls p1.mul(p2) -> ok
    print(span((5*p1).y)) #calls 5.mul(p2) -> falls back on p2.__rmul__
    print(span((p1*5).y)) #calls p1.mul(5) -> ok
    

def test_profile_mul2(p1,res):    

    test_plot(p1*res) #calls p1.mul(p2) -> ok
    test_plot(res*p1)
    test_plot(p1*p1)
    test_plot(p1*5) #calls p1.mul(p2) -> ok
    test_plot(5*p1)
    test_plot(5*res)
    test_plot(res*5) #calls p1.mul(5) -> ok

def test_profile_class():

    #test hardcore labels
    from pyProfile.profile_class import test_profile_mul2

    res=p1*2
    res.name = ''
    res.units = ['keV','']

    test_profile_mul2(p1,p2)
    test_profile_mul2(p1,res)
    test_profile_mul2(res,res)

    #test hardcore labels
    from pyProfile.profile_class import test_profile_div2

    res=p1*2
    res.name = ''
    res.units = ['keV','']

    test_profile_div2(p1,p2)
    test_profile_div2(p2,res)
    test_profile_div2(res,res)

def test_plot(p1):
    plt.figure()
    (p1).plot()
    plt.legend()
    print(p1)
    return p1


    
def test_profile_div(p1,p2):    
    
    test_profile_division(p1,p2)
    
    #test div
    print(span(p1.y))
    print('---------------')
    print(span((p1/p2).y)) #calls p1.div(p2) -> ok
    print(span((5/p1).y)) #calls 5.div(p2) -> falls back on p2.__rdiv__
    print(span((p1/5).y)) #calls p1.div(5) -> ok
    
    test_plot(p1)
    test_plot(p1/p2) #calls p1.mul(p2) -> ok
    test_plot(5/p1)
    test_plot(p1/5) #calls p1.mul(5) -> ok
 
def test_profile_div2(p1,res):    

    test_plot(p1/res) #calls p1.mul(p2) -> ok
    test_plot(res/p1)
    test_plot(p1/p1)
    test_plot(p1/5) #calls p1.mul(p2) -> ok
    test_plot(5/p1)
    test_plot(5/res)
    test_plot(res/5) #calls p1.mul(5) -> ok 
    

# test profile division
def test_profile_division(p1,p2):
    """ plot division of profile 2 by 1 and inverse of the inverse ratio based on np only and compare with division between objects, interpolate over 1st. Plot comparison with what is calculated by np only."""
    
    def divide (x1,y1,x2,y2):
        """division based only on np."""
        yy2=np.interp(x1,x2,y2)
        yr= y1/yy2
        return yr
    
    x1,y1 = p1()
    x2,y2 = p2()

    r = p2/p1
    plt.figure()
    
    r.plot()
    plt.plot(x2,divide(x2,y2,x1,y1),label='ratio p2/p1')
    plt.plot(x1,1/divide(x1,y1,x2,y2),label='1/ratio p1/p2')
    plt.legend()
    plt.xlim([0.1,14])
    plt.ylim([0.5,1.65])

if __name__=='__main__':
    plt.close('all')
    test_class_init()
