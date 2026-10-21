
"""
Extensible vectorizable list.

In final version has list properties `iterators`,
`aggregator`,`expanded` containing methods that are vectorized in different ways. Iterator is default, return a list with each element obtained by applying its method with same name to each element of `superlist`. 
Aggregator accumulate a binary operation, returning a single value. Expanded return a single element by passing the content of `superlist` as n elements to the element method. 
Note that passing parameters must be handled accordingly. In particular a list of parameters or a single value can be passed to iterators, however this can create ambiguity. For example, an element method can accept list or single values, so it is not clear when a list is passed to superlist, if this means that the same list is passed to all elements as list argument of element method, rather than passing each element of list argument as scalar argument of elements method. There is no ambiguity if a list of lists or a list with len different from number of superlist elements, this should be checked, otherwise a warning should be visualized telling about the possible ambiguity and default action (possibly passing one value per element, call with nested list [[element]] to apply as list to all values).

In this experimental version, different ways to access object methods are tested.

functions operating on a list of Data2D objects

Another problem is to make sure that superlists are returned on list operators like slices, how to do this? Metaclasses?

2022/07/12 after last changes it seems to work quite well. 
TODO: implement assignment, e.g. pl.x = 12.398/pl.x

2020/05/26 moved to dataIO"""

# turned into a class derived from list 2020/01/16, no changes to interface,
# everything should work with no changes.

import numpy as np
#from dataIO.functions import prep_kw  # to be imported from outside
#from dataIO.dicts import pop_kw

class xSuperlist(list):
    """A list of pySurf.Data2D objects on which unknown operations are performed serially."""    
    # property returns a function, which if called returns a list with the property of each item
    # method returns a list with the methods of each item as function. 
    
    def __getattr__(self,name,*args,**kwargs): 
            # devo costruire una nuova funzione che preso un oggetto
            # lista ritorna un oggetto lista ottenuto dal valore restituito dalla funzione su ogni elemento.
            def newfunc(*args, **kwargs):
                attr = [object.__getattribute__(name) for object in self]
                result = [a(*args, **kwargs) if hasattr(attr, '__call__') else a for a in attr]
                return result
            return newfunc
            
class xxSuperlist(list):
    """Working version from Superlist6. """   
    # A problem is that a list is returned, not a superlist, so for example calling
    #   s.crop([2,10]) returns a list of objects, but I cannot apply a second time about
    #   method. 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getattr__(self,name,*args,**kwargs): 
        attr = [object.__getattribute__(name) for object in self]
        
        if hasattr(attr[0], '__call__'):
            # builds a function that, taken a list object oggetto
            # return a list of the results obtained applying function to each item.
            def newfunc(*args, **kwargs):
                attr = [object.__getattribute__(name) for object in self]
                result = [a(*args, **kwargs) for a in attr]
                return result
            return newfunc
        else:
            # return list of attributes
            return attr
    
    def __getattribute__(self,name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                result = attr(*args, **kwargs)
                return result
            return newfunc
        else:
            return attr
            

## DEVELOPMENT VERSIONS

class superlist1(list):
    """Test class that vectorizes methods. Basic, implements properties,
        but not methods."""
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        result = []
        if hasattr(attr, '__call__'):   #try to identify a method as a callable attr
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                print('before calling %s' %a.__name__)
                result = a(*args, **kwargs)
                print('done calling %s' %a.__name__)
                return result
            return newfunc
        else:
            return attr

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.

class superlist2(list):
    """Test class that vectorizes methods."""
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        """
        if hasattr(attr[0], '__call__'):
            attr=attr[0]
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return [a for a in attr]
        """
        
        result = []
        for a in attr:
            #print('loop ',a)
            #pdb.set_trace()
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
                """
                def newfunc(*args, **kwargs):
                    #pdb.set_trace()
                    print('before calling %s' %a.__name__)
                    result = a(*args, **kwargs)
                    print('done calling %s' %a.__name__)
                    return result
                result.append(newfunc)
                """
            else:
                result.append(a)

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.
        
class superlist3(list):
    """Test class that vectorizes methods.  """
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        result = []
        for a in attr:
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
            else:
                result.append(a)

        return result  #it works as a property. As method, this returns a list of methods, but since the method is called as sl.method(),
        #gives an error as it tries to call the list.
        # deve ritornare una funzione che ritorna una lista.  

class superlist4(list):
    """Test class that vectorizes methods.  """
    
    def __getattr__(self,name):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = [obj.__getattr__(self, name) for obj in self] #questo non funziona
        attr = [object.__getattribute__(name) for object in self]
        
        result = []
        for a in attr:
            if hasattr(a, '__call__'):
                result=np.vectorize(a)
            else:
                result.append(a)

        return superlist4(result)  #deve restituire un istanza della classe
                #stessa, perche' se ritorna lista non posso applicare ulteriori
                #metodi.

class superlist_alt(list):
    """2021/07/14 da https://stackoverflow.com/questions/2704434/intercept-method-calls-in-python anche testata in Dlist."""
    
    def __getattribute__(self,name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return attr
            
class superlist5(list):
    """2021/07/14 da Dlist, versione funzionante con proprieta' e metodi, portata qui e non testata."""
    
    def __getattr__(self,name,*args,**kwargs):  #originariamente usava __getattribute__, che riceve attributo
        # prima di chiamarlo (quindi anche se gia' esistente).
        #attr = object.__getattr__(self, name) #questo non funziona
        
        attr = [object.__getattribute__(name) for object in self]
        print("ATTR:",attr,*args,**kwargs)
        print("SELF,name",self,name)
        
        if hasattr(attr[0], '__call__'):
            print("call methods of items")
            # devo costruire una nuova funzione che preso un oggetto
            # lista ritorna un oggetto lista ottenuto dal valore restituito dalla funzione su ogni elemento.
            def newfunc(*args, **kwargs):
                attr = [object.__getattribute__(name) for object in self]
                result = [a(*args, **kwargs) for a in attr]
                return result
            return newfunc
            
            """
            # per quale motivo restituire una funzione?

                if hasattr(attr[0], '__call__'):
                    return [a(*args,**kwargs) for a in attr]  # e' una generica lista 
                else:
                    # return list of attributes
                    return Dlist(attr) #attr

            Il codice quasi funziona, nel senso che genera plot, ma da anche errore. In realta' chiamando D.plot  (senza parentesi), la lista viene restituita comunque.
            Questo perche' __getattr__ viene chiamato da qualcosa che si aspetta di vedere restituita una funzione, non un valore.

            Per questo l'equivalente unidimensionale era:
            
            def newfunc(*args, **kwargs):
                #pdb.set_trace()
                #print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                #print('done calling %s' %attr.__name__)
                return result
            return newfunc
            """
        else:
            # return list of attributes
            print("return atribute of items")
            return superlist5(attr) #attr  #this can also be list. using superlist anyway makes it possible to call item methods also on results.
            # example, I can call shape of shapes from nested arrays. [[1,2],[1,2,3],[4,5]].shape.shape -> [2,2,2] (not working)
            
class Superlist6(list):
    """Working version for debug of getattr/getattribute.
    print messages."""    
    
    def __getattr__(self,name,*args,**kwargs): 
        print("GETATTR!")
        attr = [object.__getattribute__(name) for object in self]
        print("ATTR:",attr,*args,**kwargs)
        print("SELF,name",self,name)
        
        if hasattr(attr[0], '__call__'):
            print("call methods of items")
            # devo costruire una nuova funzione che preso un oggetto
            # lista ritorna un oggetto lista ottenuto dal valore restituito dalla funzione su ogni elemento.
            def newfunc(*args, **kwargs):
                attr = [object.__getattribute__(name) for object in self]
                result = [a(*args, **kwargs) for a in attr]
                return result
            return newfunc
        else:
            # return list of attributes
            print("return atribute of items")
            return attr
    
    def __getattribute__(self,name):
        print("GETATTRIBUTE!")
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                print('before calling %s' %attr.__name__)
                result = attr(*args, **kwargs)
                print('done calling %s' %attr.__name__)
                return result
            return newfunc
        else:
            return attr

class Superlist(list):
    """Working version from Superlist6. Returns a superlist."""   
    # A problem was that if a list is returned, not a superlist, for example calling
    #   s.crop([2,10]) returns a list of objects, but I cannot apply a second time about
    #   method. 
    
    
    def __getattr__(self,name,*args,**kwargs): 
        attr = [object.__getattribute__(name) for object in self]
        
        if hasattr(attr[0], '__call__'):
            # builds a function that, taken a list object oggetto
            # return a list of the results obtained applying function to each item.
            def newfunc(*args, **kwargs):
                attr = [object.__getattribute__(name) for object in self]
                # 2023/01/17 replaced:
                #result = Superlist()
                #result.extend([a(*args, **kargs) for a in attr])
                # which was returning a Superlist, now returns the right type (e.g. Dlist).
                result = self.__class__([a(*args, **kwargs) for a in attr])
                return result
            return newfunc
        else:
            # return list of attributes
            return attr
    
    def __getattribute__(self,name):
        attr = object.__getattribute__(self, name)
        if hasattr(attr, '__call__'):
            def newfunc(*args, **kwargs):
                #breakpoint()
                result = attr(*args, **kwargs)
                return result
                #return self.__class__(result)  #doesn't work, big crash
            return newfunc
        else:
            return attr
        
    def __getitem__(self, index):
        """This implements slicing, according to chatGPT suggestions."""
        
        result = super().__getitem__(index)
        if isinstance(result, list):
            return self.__class__(result) #Superlist(result)
        return result


def test_superlist_dlist(D):
    #create a dlist D
    # test
    print('dlist:')
    print(D)
    print('\ntest property (D.name):')
    print(D.name)
    print('\ntest method (D.plot):')
    print(D.plot())
    
    # existing method
    D.plot

    # existing method call
    D.plot()
      
    # non existing 
    D.cane


         
def test_superlist(cls,obj=None):

    print ('test class ',cls)
    if obj is None:
        #s=type(cls,[np.arange(4),np.arange(3)])
        s = cls([np.arange(4),np.arange(3)])
        #s = superlist([np.arange(4),np.arange(3)])

    print('original data:')
    print(s)
    print('\ntest property (np.shape):')
    print(s.shape)
    print('\ntest method (np.flatten):')
    print(s.flatten())
    return s
    
    
if __name__ == "__main__":
    s = test_superlist(Superlist)
        #test_superlist(superlist)
    """
    except: # catch *all* exceptions
        e = sys.exc_info()[0]
        print( "Error: %s" % e )
    """