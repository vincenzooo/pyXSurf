import numpy as np
from scipy.optimize import minimize
from pySurf.points import *
from dataIO.fn_add_subfix import fn_add_subfix

def closest_point_on_line(points,lVersor,lPoint):
    """From a list of points in 3D space as Nx3 array, returns a Nx3 array with the corresponding closest points on the line."""
    #vd=lVersor/((np.array(lVersor)**2).sum())  #normalize vector, not sure it is necessary.
    vd=lVersor/np.sqrt(np.sum(np.array(lVersor)**2))  #normalize vector, not sure it is necessary.
    return lPoint+(points-lPoint).dot(vd)[:,np.newaxis]*(vd)

'''
def closest_point(points,reference):
    """From a list of points in 3D space as Nx3 array, returns the closest one to reference point."""

    dist=np.abs(np.array(points)-reference))
    
    return points[nanargmin([:,2]
'''
    
def cylinder_error(odr=(0,0,0,0,0,0),points=None,retall=False,extra=False):  #origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit
    """Given a set of N points in format Nx3, returns the error on the cylinder defined by origin and direction as a 6(3+3) dim vector.
    If retall is set, additional values are returned :
        radius: best fit radius for the cylinder.
        deltaR[N,3]: deviation from radius for each point.
    extra is equivalent to retall, renamed 2018/02/15 kept for backward compatibility"""
    
    #ca 400 ms per loop
    origin=odr[0:3]
    direction=odr[3:]
    #vd=direction/((np.array(direction)**2).sum())  #normalize vector, not sure it is necessary.
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,direction,origin) #points on the cylinder axis closer to points to fit
    deltaR2=((points-Paxis)**2).sum(axis=1)-radius**2  #square difference from expected radius for each point
    fom=np.sqrt(deltaR2.sum()/len(deltaR2))
    retall=retall | extra
    if retall:
        residuals=np.hstack([x,y,deltaR[:,None]])
        return fom,residuals,radius
    else: return fom  #,deltaR,radius   

def cylinder_error3(odr:float=(0,0,0,0),points:float=None,radius:float=None,retall:bool=False,extra:bool=False,xy:bool=False)->float:  
#from v1
#origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit

    """
    Calculate rms surface error for points wrt a cylinder defined by axis vector.

    Given a set of N points in format Nx3, returns the error on the cylinder defined by origin and direction of axis. Best fit radius is calculated as best fit if not provided. Otherwise the provided value is used with no fit.
    nan are not allowed and must be filtered before calling the function (this is more efficient if the function is called many times, e.g. in optimization).
    Args:
        points : points in format Nx3
        odr : 4-element axis vector in format (origin_y,origin_z,direction_x,direction_z)
        radius (optional) : If provided, radius is used, otherwise best fit radius for the given axis is calculated.
        retall (optionall): if set, additional values are returned :
            radius: best fit radius for the cylinder.
            deltaR[N,3]: deviation from radius for each point.
    extra (optional) : equivalent to retall, renamed 2018/02/15 kept for backward compatibility.

    Returns:
        rms of radial distance of points from ideal cylindrical surface.
    """
    
    #2015/09/05 added functions for nan from np
    #2015/08/26 changed sign of deltaR to follow bump up convention (and cone_error3)
    #ca 400 ms per loop
	#get a point on plane y=0 unless xy is set, then plan is x=0.
	#same for origin is vector out of origin plane.
    if xy:
        origin=(0,odr[0],odr[1])
        direction=(1.,odr[2],odr[3])
    else:
        origin=(odr[0],0,odr[1])
        direction=(odr[2],1.,odr[3])
    
    vd=direction/np.sqrt(1+(np.array(direction)**2).sum())  #normalize vector
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,vd,origin)
    #origin+(points-origin).dot(vd)[:,np.newaxis]*(vd) #points on the cylinder axis closer to points to fit
    R=np.sqrt(np.nansum(((points-Paxis)**2),axis=1)) #distance of each point from axis
    #R=np.sqrt(((points-Paxis)**2).sum(axis=1)) #distance of each point from axis
    if radius is None:
        radius=np.nanmean(R)  #
        #radius=R.mean()  #
    #plt.plot(R-radius)
    deltaR=(radius-R)  #difference from expected radius for each point
    fom=np.sqrt((np.nansum((deltaR)**2))/len(deltaR))
    #fom=np.sqrt(((deltaR)**2).sum()/len(deltaR))
    residuals=np.hstack([x,y,deltaR[:,None]])
    retall=retall | extra
    if retall: return fom,residuals,radius
    else: return fom  #,deltaR,radius

def cone_error(odr=(0,0,0,0,0,0),points=None,retall=False,extra=False):  #origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit
    """Given a set of N points in format Nx3, returns the rms surface error on the cone defined by origin (intercept of the axis with x=0) and direction, 
    passed as 4-vector odr(origin_y,origin_z,direction_x,direction_z). 
    Best fit cone for odr is calculated from linear fit of data. 
    If retall is set, additional values are returned :
    coeff: best fit radius for the cone as [m,q] for x' distance from x=0 plan on cone axis R(x')=m x' + q. Half cone angle is atan(m). 
    deltaR[N,3]: deviation from radius for each point.
    extra is equivalent to retall, renamed 2018/02/15 kept for backward compatibility
    """
    #ca 400 ms per loop
    origin=odr[0:3]
    direction=odr[3:]
    vd=direction/np.sqrt((1+np.array(direction)**2).sum())  #normalize vector, not sure it is necessary.
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,direction,origin)
    Paxdist=np.sqrt(((Paxis-origin)**2).sum(axis=1)) #distance of each point from
    R=np.sqrt(((points-Paxis)**2).sum(axis=1)) #distance of each point from axis
    coeff=np.polyfit(Paxdist,R,1) #best fit cone
    deltaR=R-coeff[0]*Paxdist-coeff[1] #residuals
    fom=np.sqrt(((deltaR)**2).sum()/len(deltaR))
    residuals=np.hstack([x,y,deltaR[:,None]])
    retall=retall | extra
    if retall: return fom,residuals,coeff
    else: return fom  #,deltaR,radius

def cone_error3(odr=(0,220.,0,0),points=None,coeff=None,retall=False,extra=False):
#2015/09/05 added functions for nan from np
#2015/01/13 changed sign of deltaR to set bump-positive (it was opposite)
#from v1
#origin=(0,0,0),direction=(0,1.,0),radius is calculated from best fit
    """Given a set of N points in format Nx3, returns the rms surface error on the cone defined by 
    its axis (radius and apex are determined by best fit).
    Axis is defined as a 4 elements vector odr=(x,z,cx,cz), not in xz plane.
    origin (intercept of the axis with y=0) and director cosines.
    If coeff is passed as input, the fit for cone surface is not performed and the coeff values are used.
    If retall is set, additional values are returned :
    coeff: best fit radius for the cone as [m,q] for x' distance from x=0 plan on cone axis R(x')=m x' + q. Half cone angle is atan(m). 
    deltaR[N,3]: deviation from radius for each point. Bump positive convention (smaller radius is positive).
    extra is equivalent to retall, renamed 2018/02/15 kept for backward compatibility
    """
    
    #pdb.set_trace()
    points=points[~np.isnan(points[:,2]),:]
    
    #ca 400 ms per loop
    origin=np.array((odr[0],0,odr[1])) #point of axis at y=0
    direction=np.array((odr[2],np.sqrt(1-odr[2]**2-odr[3]**2),odr[3])) # director cosines, cy is assumed positive
    #direction=direction/np.sqrt((direction**2).sum())
    direction=direction/(np.sqrt(np.nansum(direction**2))) #normalize
    x,y,z=np.hsplit(points,3)
    Paxis=closest_point_on_line(points,direction,origin)
    Paxdist=np.sqrt(np.nansum((Paxis-origin)**2,axis=1))*np.sign(Paxis[:,1]-origin[1])
    #Paxdist=np.sqrt(((Paxis-origin)**2).sum(axis=1))*np.sign(Paxis[:,1]-origin[1])
    
    #d: on axis coordinate of points
    R=np.sqrt(np.nansum((points-Paxis)**2,axis=1)) #r: distance of each point from axis
    #R=np.sqrt(((points-Paxis)**2).sum(axis=1)) #r: distance of each point from axis
    if coeff is None:
        coeff=np.polyfit(Paxdist,R,1) #best fit cone
    deltaR=coeff[0]*Paxdist+coeff[1]-R #residuals
    fom=np.sqrt(np.nansum((deltaR)**2)/len(deltaR))
    #fom=np.sqrt(((deltaR)**2).sum()/len(deltaR))
    residuals=np.hstack([x,y,deltaR[:,None]])
    retall=retall | extra
    if retall: return fom,residuals,coeff
    else: return fom  #,deltaR,radius
	
def subtract_cylinder(pp,odr,sampleName='',verbose=True):
#not used in v1
    """
    odr: 6-vector (origin_y,origin_y,origin_z,direction_x,direction_y,direction_z),
        note that  this is redundant, since only two components are enough for direction 
        (magnitude is irrelevant).
    pp: complete set of points Npx3
    """
    fom,deltaR,radius=cylinder_error(odr,pp,retall=False,extra=True)
    xymin=np.nanmin(pp,axis=0)[0:2]
    xymax=np.nanmax(pp,axis=0)[0:2]
    rp=plot_points(np.hstack([pp[:,0:2],deltaR[:,None]*1000]),shape=(281,3001)) #this is done to easily replot and change scale

    #surface map and data
    plt.clf()
    plt.imshow(rp,aspect='equal',interpolation='none',vmin=-5,vmax=10,
        extent=[xymin[1],xymax[1],xymin[0],xymax[0]])
    plt.colorbar()
    plt.title(sampleName+(sampleName if sampleName else '')+'best-fit-cylinder removed.')
    plt.xlabel('Y(mm)')
    plt.ylabel('X(mm)')
    plt.savefig(fn_add_subfix(datafile,'_cylinder','png'))
    save_points(np.hstack([pp[:,0:2],deltaR[:,None]*1000]),fn_add_subfix(datafile,'_cylinder'))    
    
    #cylinder output

    misal=np.arccos(1/np.sqrt(1+((odr[2:]**2).sum())))
    
    if verbose:
        print ('Best fit radius %s'%radius)    
        print ('Misalignment of optical axis: %s rad (%s deg)'%(misal,misal*np.pi/180))
        
        #rms output
        print ('rms entire surface %s'%(np.nanstd(rp)))
        plt.figure()
        plt.plot(np.nanstd(rp))
        plt.plot(np.nanstd(rp,axis=0))
        plt.plot(np.nanstd(rp,axis=1))

    #mask outliers on a copy
    rp2=rp[:]
    np.where(np.isnan(rp2))

def subtract_cone(pp,odr,sampleName='',outfile=None,vmin=None,vmax=None,verbose=True):
#not used in v1
    """
    odr: 6-vector (origin_y,origin_y,origin_z,direction_x,direction_y,direction_z),
        note that  this is redundant, since only two components are enough for direction 
        (magnitude is irrelevant).
    pp: complete set of points Npx3
    verbos: if set print output and generate plot
    """
    fom,deltaR,coeff=cone_error(odr,pp,retall=True)
    xymin=np.nanmin(pp,axis=0)[0:2]
    xymax=np.nanmax(pp,axis=0)[0:2]
    rp=plot_points(np.hstack([pp[:,0:2],deltaR[:,None]*1000]),shape=(281,3001)) #this is done to easily replot and change scale

    #surface map and data
    plt.clf()
    plt.imshow(rp,aspect='equal',interpolation='none',vmin=vmin,vmax=vmax,
        extent=[xymin[1],xymax[1],xymin[0],xymax[0]])
    plt.colorbar()
    plt.title(((sampleName+' - ') if sampleName else (''))+'best-fit-cone removed.')
    plt.xlabel('Y(mm)')
    plt.ylabel('X(mm)')
    if outfile:
        plt.savefig(fn_add_subfix(outfile,'_cone','png'))
        save_points(np.hstack([pp[:,0:2],deltaR[:,None]*1000]),fn_add_subfix(outfile,'_cone'))    
    
    #cone output
    m=coeff[0]
    if verbose:
        print ('Cone angle:%s+/-%s rad(%s+/-%s deg) '%(np.arctan(m),0,np.arctan(m)*180/np.pi,0))
        print ('Axis intercept at x=0: %smm '%(coeff[1]))
        misal=(np.arccos(1/np.sqrt(1+((odr[2:]**2).sum()))))
        print ('Misalignment of optical axis: %s rad (%s deg)'%(misal,misal*180./pi))
        
        #rms output
        print ('rms entire surface %s'%(np.nanstd))
        
        plt.figure()
        plt.plot(np.nanstd(rp))
        plt.plot(np.nanstd(rp,axis=0))
        plt.plot(np.nanstd(rp,axis=1))

    #mask outliers on a copy
    rp2=rp[:]
    np.where(np.isnan(rp2))   
    return fom,deltaR,coeff

"""   
def fit_cylinder(points,guessValue=None,callback=None):
#not used in v1
    if guessValue!=None:
        odr=guessValue
    else:
        odr=(0,220.,0,0)
    result=minimize(cylinder_error,x0=(odr,),args=(points,),options={'maxiter':1000},callback=callback,method='Nelder-Mead')
    d=result.x[[2,3]]
    angle=np.arccos()

def fit_cone(points,guessValue=None):
#not used in v1
    if guessValue!=None:
        odr=guessValue
    else:
        odr=(0,220.,0,0)
    result=minimize(cone_error,x0=(odr,),args=(points,),options={'maxiter':1000},callback=p,method='Nelder-Mead')
    d=result.x[[2,3]]
    angle=np.arccos()
"""

def fit_cone(pts,odr2,zscale=1.,keepnan=False,verbose=True,**kwargs):
    """   fit pts and return residuals. Info are printed. odr2 is starting guess.
    
    fom,deltaR,pars=fit_cone(pts,odr2,fom_func)
    zscale is the factor to multiply data to obtain same units as x and y, 
    e.g. 1000. for x and y in mm and z in um. Output is in same unit as input.
    pts and deltaR enter and exit with z in microns
    parameters for optimization can be passed as keyword arguments, e.g.:
    
    #evaluate fom without optimization fit_cone/fit_cylinder
    deltaR,o = fom_function(pts,odr,zscale=1000.,options={'radius':radius})
    
    # fit from starting guess value odr0
    deltaR,odr=fit_cylinder(c,odr0,zscale=1000.,options={'maxiter':500},
        method='Nelder-Mead') 
        
    typically:
    odr2=(span(xg).sum()/2.,220.,0,0.) #use nominal value for guess direction.
    
    See developer notes in code.
    """
    
    fom_func=cone_error3
    pts=pts/[1,1,zscale] #data are um, convert to mm before fit

    #filter for nans, note that this is done here rather than in fom_func so it
    #is performed only once.
    mask=~np.isnan(pts[:,2])
    
    #find odr from optimization if maxiter is set otherwise use
    # input parameters `odr`. Needs an absurde dictionary with key `options`
    #  use `fit_cone(...,options={'maxiter':500})` Will run optimization
    #  only if this is set to a number, otherwise will just evaluate
    #  the fom. It's an ugly workaround mishandling **kwargs.
    # done for not knowing it better.
    
    mi=None  #mi=maxiter if called with fit_cone(..,maxiter=int)
    if 'options' in kwargs:
        mi=kwargs['options'].pop('maxiter',None) 
    if mi:  
        result=minimize(fom_func,x0=(odr2),
                args=(pts[mask,:],),
                **kwargs)   #options={'maxiter':500},
                            #method='Nelder-Mead')#,callback=p)
        odr=result.x
    else:
        print('Dictionary `options` ')
        #if set to 0 r unset do not perform optimization
        odr=odr2
        
    s=span(pts,axis=0)    
    fom,deltaR,coeff=fom_func(odr,pts[mask,:],retall=True)
    deltaR[:,2]=deltaR[:,2]*zscale
    if keepnan:
        pts[mask,2]=deltaR[:,2]
        deltaR=pts
        
    origin=(odr[0],0,odr[1]) #point of axis at y=0
    direction=np.array((odr[2],np.sqrt(1-odr[2]**2-odr[3]**2),odr[3])) # director cosines, cy is assumed positive
    direction=direction/np.sqrt((direction**2).sum())  
    
    if verbose:
        print ('-----------------------------------')
        print ('Results of fit with function: "%s" on region:'%fom_func)
        print ('X: [%6.3f,%6.3f]'%tuple(s[0]))
        print ('Y: [%6.3f,%6.3f]'%tuple(s[1]))
        print ('data range: [%6.3f,%6.3f]'%tuple(s[2]))
        print ('---')
        if mi:
            print (result )   
    
        print ('-----------------------------------')
        print ('Axis direction, components (director cosines): ')
        print (direction)
        print ('Angle of cone axis with x,y,z axis (deg):')
        print (np.arccos(np.abs(direction))*180/np.pi)
        print ('Axis intercept with y=0: ')
        print (origin)
        print ('Cone parameters (angle(deg), R@y=0) [A,B for R=A]:')
        print (np.arctan(coeff[0])*180/np.pi,coeff[1])
        print ('---')
        print ('rms residuals=%s [units of Z]'%(fom*zscale))
        #coordinates of cone apex

    return deltaR,odr        
    
        
def fit_cylinder(pts,odr2,zscale=1.,keepnan=False,align=False,verbose=True,**kwargs):  
    """   fit pts and return residuals. Info are printed. 
    odr2: starting guess for fom func parameters. 
        for cylinder_error3, it is (origin_y,origin_z,direction_x,direction_z),
        meaning for axis nearly parallel to y: odr2=(x_center,R,0.,0.) 
        fom,deltaR,pars=fit_cylinder(pts,odr2,fom_func)
    zscale: factor to divide z data to obtain same units as x and y, 
        e.g. 1000. for x and y in mm and z in um. Output is in same unit as input.
    parameters for optimization with scipy.optimize.minimize can be passed as keyword arguments, 
        e.g.: deltaR,odr=fit_cylinder(c,odr2,options={'maxiter':500},method='Nelder-Mead')
    if keepnan is set True nans are reinserted after the fit keeping same size for array,
        otherwise nans are removed.
    if align is set True, rotate data to align axis and return residuals on rotated grid (can be resampled on data or viceversa
    with resample_points)-- NOT WORKING
    """    
    fom_func=cylinder_error3
    pts=pts/[1,1,zscale] #data are um, convert to mm before fit
    
    #filter for nans, note that this is done here rather than in fom_func so it
    #is performed only once.
    mask=~np.isnan(pts[:,2])
    
    if not isinstance(odr2,np.ndarray): odr2=np.array(odr2)
    
    #find odr from optimization if maxiter is set otherwise use 
    mi=None
    radius=None
    if 'options' in kwargs:
        mi=kwargs['options'].pop('maxiter',None)
        radius=kwargs['options'].pop('radius',None)
        
    if mi is not None:
        if mi != 0:
            result=minimize(fom_func,x0=(odr2),
                    args=(pts[mask,:]),
                    **kwargs)   #options={'maxiter':500},
                                #method='Nelder-Mead')#,callback=p)
            odr=result.x
        else:
            odr=odr2
    else:
        #if set to 0 or unset do not perform optimization
        odr=odr2
        
    s=span(pts,axis=0)    
    fom,deltaR,coeff=fom_func(odr,pts[mask,:],radius=radius,retall=True)  #calculate for best fit parameters
    deltaR[:,2]=deltaR[:,2]*zscale    #output is in same units as input
    if keepnan:
        pts[mask,2]=deltaR[:,2]
        deltaR=pts
    
    #fom_func, s, zscale, odr, fom, coeff
    vy=np.sqrt(1-(odr[-2:]**2).sum())
    az=np.arctan2(odr[2],vy)
    
    if verbose:
        print ('-----------------------------------')
        print ('Results of fit with function: "%s" on region:'%fom_func)
        print ('X: [%6.3f,%6.3f]'%tuple(s[0]))
        print ('Y: [%6.3f,%6.3f]'%tuple(s[1]))
        print ('data range: [%6.3f,%6.3f]'%tuple(s[2]))
        print ("---")
        if mi:
            print (result )      
        print ('(Angle of cyl axis with y axis (deg):')
    
        print (np.arccos(vy)*180/np.pi)
        print ('Rotation about z axis from y axis(deg):')
        print (az*180/np.pi)
        print ('Radius:')
        print (coeff)
        print ('Axis Parameters:')
        print (odr)
        print ('---')
        print ('rms residuals=%s um'%(fom*zscale))
    
    if align:
        c=span(deltaR,axis=0)[:2].sum(axis=1)/2 #center of all points
        deltaR=rotate_points(deltaR,az,center=c)
    return deltaR,odr 

    
    
if __name__=='__main__':
    fom_func=cone_error    #this is the function giving the FOM to be minimized
    def p(x): print (x) #,cylinder_error(x,points)
    outSubfix='_cone' #the name of output file is the datafile with this subfix added
    datafile=r'test\test_fitCylinder\OP2S04OP2S04b\04_OP2S04_xyscan_Height_transformed.dat'
    #datafile='OP2S03c/22_OP2S03_yxsurf_Height_transformed.dat'
    #datafile='OP2S04/04_OP2S04_xyscan_Height_transformed.dat'
    #datafile='OP2S05/05_OP2S05_xysurf_Height_transformed.dat'
    

    #create points to be fit from a subset of points.
    pts=get_points(datafile,delimiter=' ')
    #pts=rotate_points(pts,-np.pi/2)
    pts=crop_points(pts,(-28,33),(-75,65))
    pts[:,2]=pts[:,2]/1000.
    c=crop_points(pts,(-28,33),(-50,50))    #[0:-1:1000,:]
    odr2=(33,220.,0,0,220)
    result=minimize(fom_func,x0=(odr2[0:-1],),args=(c,),options={'maxiter':1000},callback=p,method='Nelder-Mead')
    print ('-----------------------------------')
    print ('Results of fit on subset of points:')
    print (result)
    
    #create output results applying the value from fit to all points
    odr=result.x
    fom,deltaR,coeff=fom_func(odr,pts,retall=True)
    origin=(odr[0],0,odr[1])
    direction=(odr[2],1.,odr[3])
    deltaR[:,2]=deltaR[:,2]*1000
    print ('-----------------------------------')
    print ('Results of fit applied to complete set of points:')
    print ('F.O.M.=%s'%(fom))
    
    plot_points(deltaR,vmin=-5,vmax=10,scatter=1,aspect='equal')
    save_points(deltaR,filename=fn_add_subfix(datafile,outSubfix))
    plt.savefig(fn_add_subfix(datafile,outSubfix,'.png'))

    