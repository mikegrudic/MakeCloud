#!/usr/bin/env python
"""                                                                            
MakeCloud: "Believe me, we've got some very turbulent clouds, the best clouds. You're gonna love it.

Usage: MakeCloud.py [options]

Options:                                                                       
   -h --help            Show this screen.
   --R=<pc>             Outer radius of the cloud in pc [default: 100.0]
   --Rmin=<pc>          Inner radius in pc if applicable [default: 0.0]
   --M=<msun>           Mass of the cloud in msun [default: 1e6]
   --filename=<name>    Name of the IC file to be generated
   --N=<N>              Number of gas particles [default: 125000]
   --MBH=<msun>         Mass of the central black hole [default: 0.0]
   --S=<f>              Rotational energy as a fraction of binding energy [default: 0.0]
   --turb_type=<s>      Type of initial turbulent velocity (and possibly density field): grid, meshless, solenoidal or full [default: gaussian]
   --turb_seed=<N>      Which pre-generated random turbulent field to use [default: 1]
   --alpha_turb=<f>     Turbulent energy as a fraction of the binding energy [default: 1.]
   --bturb=<f>          Magnetic energy as a fraction of the binding energy [default: 0.01]
   --minmode=<N>        Minimum populated turbulent mode for Gaussian initial velocity field [default: 2]
   --turb_index=<N>     Power-law index of turbulent spectrum for Gaussian random initial velocity [default: 2]
   --turb_meshless      Use new method for generating ~k^-2 velocity power spectrum without a grid (SLOW)
   --poisson            Use random particle positions instead of a gravitational glass
   --poisson_seed=<N>   Random seed for generating particle positions, if --poisson is used [default: 42]
   --turb_path=<name>   Contains the root path of the turb [default: /panfs/ds08/hopkins/mgrudic/turb]
   --glass_path=<name>  Contains the root path of the glass ic [default: /home/mgrudic/glass_orig.npy]
   --G=<f>              Gravitational constant in code units [default: 4.3e4]
   --warmgas=<f>        Add warm ISM envelope with total mass equal to this fraction of the nominal mass [default: 0.0]
"""

import numpy as np
from scipy import fftpack, interpolate, ndimage
from scipy.integrate import quad, odeint, solve_bvp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import sys
import h5py
from numba import jit, prange
from docopt import docopt
from pykdgrav import Potential

arguments = docopt(__doc__)
R = float(arguments["--R"])/1e3
rmin = float(arguments["--Rmin"])/1e3
M_gas = float(arguments["--M"])/1e10
N_gas = int(float(arguments["--N"])+0.5)
M_BH = float(arguments["--MBH"])/1e10
spin = float(arguments["--S"])
turbulence = float(arguments["--alpha_turb"])
turb_type = arguments["--turb_type"]
turb_seed = int(float(arguments["--turb_seed"])+0.5)
magnetic_field = float(arguments["--bturb"])
poisson = arguments["--poisson"]
turb_meshless = arguments["--turb_meshless"]
#scturb = arguments["--scturb"]
seed = int(float(arguments["--poisson_seed"])+0.5)
turb_index = float(arguments["--turb_index"])
minmode = int(arguments["--minmode"])
filename = arguments["--filename"]
turb_path = arguments["--turb_path"]
glass_path = arguments["--glass_path"]
G = float(arguments["--G"])
warmgas = float(arguments["--warmgas"])
res_effective = int(N_gas**(1.0/3.0)+0.5)

if filename==None:
    filename = "M%g_"%(1e10*M_gas) + ("MBH%g_"%(1e10*M_BH) if M_BH>0 else "") + "R%g_S%g_T%g_B%g_Res%d_n%d"%(R*1e3,spin,turbulence,magnetic_field,res_effective,minmode) +  ("_%d"%turb_seed) + ".hdf5"
    filename = filename.replace("+","").replace('e0','e')

mgas = np.repeat(M_gas/N_gas, N_gas)

if turb_type=='full':
#    ft = h5py.File("/panfs/ds08/hopkins/mgrudic/turb/supersonic/snapshot_%s_c.hdf5"%(str(turb_seed).zfill(3)))
    ft = h5py.File("/panfs/ds08/hopkins/mgrudic/turb/nomhd/snapshot_002_c.hdf5")
    x = np.float64(np.array(ft["PartType0"]["Coordinates"]))-0.5
    r = np.sum(x**2,axis=1)**0.5
    if "MagneticField" in ft["PartType0"].keys():
        B = np.float64(np.array(ft["PartType0"]["MagneticField"]))[r<0.5]
    else:
        B = 0*x
#    theta = 2*np.pi*x
#    circular_mean = np.angle(np.average(np.exp(1j*theta),axis=0))%(2*np.pi)
    v = np.float64(np.array(ft["PartType0"]["Velocities"]))[r<0.5]
    h = np.float64(np.array(ft["PartType0"]["SmoothingLength"]))[r<0.5]
    m = np.float64(np.array(ft["PartType0"]["Masses"]))[r<0.5]
    x = x[r<0.5]
    xchoice = np.random.choice(np.arange(len(x)),size=N_gas,replace=False)
    x, v, B, h, m = x[xchoice], v[xchoice], B[xchoice], h[xchoice], m[xchoice]
    plasma_beta = (0.5*np.sum(m*np.sum(v**2,axis=1)))/np.sum(np.sum(B**2,axis=1)*(4*np.pi/3*h**3/32)/(8*np.pi))
    x = x*2*R
    h = h*2*R
    #B /= (0.5/(2*R))**1.5
    r = np.sum(x**2,axis=1)**0.5
else:
    x = 2*(np.load(glass_path)-0.5)
    Nx = len(x)
    if len(x)*np.pi*4/3 / 8 < N_gas:
        x = 2*(np.load("/home/mgrudic/glass.npy")-0.5)
        
    r = np.sum(x**2, axis=1)**0.5
    x = x[r.argsort()][:N_gas]
    x *= (float(Nx) / N_gas * 4*np.pi/3 / 8)**(1./3)*R
    
    r = np.sum(x**2,axis=1)**0.5

    x, r = x/r.max(), r/r.max()
    
    rho_form = lambda r: 1. #change this function to get a different radial density profile; normalization does not matter as long as rmin and rmax are properly specified
    rho_norm = quad(lambda r: rho_form(r) * 4 * np.pi * r**2, rmin, R)[0]
    rho = lambda r: rho_form(r) / rho_norm

    rnew = odeint(lambda rphys, r3: np.exp(r3)/(4*np.pi*np.exp(rphys)**3*rho(np.exp(rphys))), np.log(R), np.log(r[::-1]**3), atol=1e-12, rtol=1e-12)[::-1,0]
    rnew = np.exp(rnew)
    x=(x.T * rnew/r).T
    r = np.sum(x**2, axis=1)**0.5
    x, r = x[r.argsort()], r[r.argsort()]
    
    if turb_type=='gaussian':
        vt = np.load(turb_path + "/gaussian/vturb%d_n%d.npy"%(minmode,turb_index))
        xgrid = np.linspace(-R,R,vt.shape[0])
        v = []
        for i in xrange(3):
            v.append(interpolate.interpn((xgrid,xgrid,xgrid),vt[:,:,:,i],x))
        v = np.array(v).T
    elif turb_type=='solenoidal':
        ft = h5py.File("/panfs/ds08/hopkins/mgrudic/turb/subsonic/snapshot_004.hdf5")#turb_seed)
        xt = 2*R*(np.array(ft["PartType0"]["Coordinates"])-.5)
        vt = np.array(ft["PartType0"]["Velocities"])
        print "getting subsonic turbulent field"
        v = vt[cKDTree(xt).query(x)[1]]

Mr = M_BH + mgas.cumsum()
ugrav = G * np.sum(Mr/ r * mgas)
#print ugrav
E_rot = spin * ugrav
I_z = np.sum(mgas * (x[:,0]**2+x[:,1]**2))
#omega = (2*E_rot/I_z)**0.5
omega = spin * np.sqrt(G*(M_BH+M_gas)/R**3)

v -= np.average(v,axis=0)
Eturb = 0.5*M_gas/N_gas*np.sum(v**2)
v *= np.sqrt(turbulence*ugrav/Eturb)

v += np.cross(np.c_[np.zeros_like(omega),np.zeros_like(omega),omega], x)


if magnetic_field>0.0 and turb_type != 'full':
    B = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.ones(N_gas)]
    uB = np.sum(np.sum(B*B, axis=1) * 4*np.pi*R**3/3 /N_gas * 3.09e21**3)* 0.03979 *5.03e-54
    B = B * np.sqrt(magnetic_field*ugrav/uB)

v = v - np.average(v, axis=0)
x = x - np.average(x, axis=0)
#print mgas, x, v, B

if turb_type=='full':
    #print(np.sqrt(turbulence*ugrav/Eturb))
    beta = (np.sum(mgas*np.sum(v**2,axis=1))*0.5)/(np.sum(np.sum(B**2,axis=1)*(4*np.pi/3*h**3/32))/(8*np.pi))
    B *= np.sqrt(beta/plasma_beta) #np.sqrt(turbulence*ugrav/Eturb)

if warmgas:
    N_warm = int(warmgas*N_gas+0.5)
    sigma_warm = 2*R*10*warmgas**(1./3)
    print(sigma_warm, R)
    x_warm = np.random.normal(size=(N_warm,3))*sigma_warm
    r_warm = np.sum(x_warm**2,axis=1)**0.5
    x = np.concatenate([x, x_warm])
    v = np.concatenate([v, np.zeros((N_warm,3))])
    Bmag = np.average(np.sum(B**2,axis=1))**0.5
    B = np.concatenate([B, Bmag * np.exp(-r_warm**2/(2*sigma_warm**2))[:,np.newaxis] * np.array([0,0,1])])
    mgas = np.concatenate([mgas, np.repeat(M_gas/N_gas,N_warm)])
else:
    N_warm = 0

import meshoid
#M = meshoid.meshoid(x,mgas)
#rho = np.repeat( / (4*np.pi*R**3/3) #32*mgas/(4*np.pi*h**3/3)
#rho, h = M.Density(), M.h
print "Writing snapshot..."
F=h5py.File(filename, 'w')
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [N_gas+N_warm,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["NumPart_Total"] = [N_gas+N_warm,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["MassTable"] = [M_gas/N_gas,0,0,0,0, M_BH]
F["Header"].attrs["BoxSize"] = 1e6
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas)
F["PartType0"].create_dataset("Coordinates", data=x)
F["PartType0"].create_dataset("Velocities", data=v)
F["PartType0"].create_dataset("ParticleIDs", data=np.arange(N_gas+N_warm)+(1 if M_BH>0 else 0))
F["PartType0"].create_dataset("InternalEnergy", data=np.ones(N_gas+N_warm))
#F["PartType0"].create_dataset("Density", data=rho)
#F["PartType0"].create_dataset("SmoothingLength", data=h)
if magnetic_field > 0.0:
    F["PartType0"].create_dataset("MagneticField", data=B)
if M_BH > 0:
    F.create_group("PartType5")
    F["PartType5"].create_dataset("Masses", data=[M_BH,])
    F["PartType5"].create_dataset("Coordinates", data=[[0,0,0]])
    F["PartType5"].create_dataset("Velocities", data=[[0,0,0]])
F.close()
