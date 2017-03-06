#!/usr/bin/env python
"""                                                                                                                                                                                     
MakeCloud: "Believe me, we've got some very turbulent clouds, the best clouds. You're gonna love it."

Usage:                                                                                                                                                                                  
MakeCloud.py [options]

Options:                                                                                                                                                                                
   -h --help         Show this screen.
   --R=<pc>          Outer radius of the cloud in pc [default: 1000.0]
   --Rmin=<pc>       Inner radius in pc if applicable [default: 0.0]
   --M=<msun>        Mass of the cloud in msun [default: 1e10]
   --filename=<name> Name of the IC file to be generated
   --N=<N>           Number of gas particles [default: 125000]
   --MBH=<msun>      Mass of the central black hole [default: 0.0]
   --S=<f>           Spin of the cloud, as a fraction of its Keplerian velocity sqrt(G M(<r) / r) [default: 1.0]
   --turb=<f>        Turbulent energy as a fraction of the Keplerian kinetic energy at a given radius [default: 0.1]
   --bturb=<f>       Magnetic energy as a fraction of the Keplerian kinetic energy at a given radius [default: 0.01]
   --minmode=<N>     Minimum populated turbulent mode [default: 4]
   --turb_index=<N>  Power-law index of turbulent spectrum [default: 2]
   --poisson         Use random particle positions instead of a gravitational glass
   --seed=<N>        Random seed for generating particle positions, if --poisson is used [default: 42]
"""

import numpy as np
from scipy import fftpack, interpolate, ndimage
from scipy.integrate import quad, odeint, solve_bvp
import sys
import h5py
from docopt import docopt

arguments = docopt(__doc__)
R = float(arguments["--R"])/1e3
rmin = float(arguments["--Rmin"])/1e3
M_gas = float(arguments["--M"])/1e10
N_gas = int(float(arguments["--N"])+0.5)
M_BH = float(arguments["--MBH"])/1e10
S = float(arguments["--S"])
turbulence = float(arguments["--turb"])
magnetic_field = float(arguments["--bturb"])
poisson = arguments["--poisson"]
seed = int(float(arguments["--seed"])+0.5)
turb_index = float(arguments["--turb_index"])
minmode = int(arguments["--minmode"])
filename = arguments["--filename"]

res_effective = int(N_gas**(1.0/3.0)+0.5)

if filename==None:
    filename = "M%g_"%(1e10*M_gas) + ("MBH%g_"%(1e10*M_BH) if M_BH>0 else "") + "R%3.2g_S%g_T%g_B%g_Res%d"%(R*1e3,S,turbulence,magnetic_field,res_effective) + ".hdf5"
    filename = filename.replace("+","").replace('e0','e')

def TurbVelField(coords, res):
    vt = np.load("/home/mgrudic/turb/vturb%d_n%d.npy"%(minmode,turb_index))
    x = np.linspace(-R,R,vt.shape[0])
    v = []
    for i in xrange(3):
        v.append(interpolate.interpn((x,x,x),vt[:,:,:,i], coords))
    return np.array(v).T

def TurbBField(coords, res):
    vt = np.load("/home/mgrudic/turb/bturb%d_n%d.npy"%(minmode, turb_index))
    x = np.linspace(-R,R,vt.shape[0])
    v = []
    for i in xrange(3):
        v.append(interpolate.interpn((x,x,x),vt[:,:,:,i], coords))
    return np.array(v).T

G = 4.3e4


if poisson:
    np.random.seed(seed)
    x = 2*(np.random.rand(2*N_gas, 3)-0.5)
else:
    x = 2*(np.load("/home/mgrudic/scripts/MakeCloud/glass_orig.npy")-0.5)
Nx = len(x)
r = np.sum(x**2, axis=1)**0.5
x = x[r.argsort()][:N_gas]

x *= (float(Nx) / N_gas * 4*np.pi/3 / 8)**(1./3)
x = x - np.average(x,axis=0)
r = np.sum(x**2, axis=1)**0.5
x, r = x[r.argsort()], r[r.argsort()]

x, r = x/r.max(), r/r.max()

rho_form = lambda r: 1. #change this function to get a different radial density profile; normalization does not matter as long as rmin and rmax are properly specified
rho_norm = quad(lambda r: rho_form(r) * 4 * np.pi * r**2, rmin, R)[0]
rho = lambda r: rho_form(r) / rho_norm

rnew = odeint(lambda rphys, r3: np.exp(r3)/(4*np.pi*np.exp(rphys)**3*rho(np.exp(rphys))), np.log(R), np.log(r[::-1]**3), atol=1e-12, rtol=1e-12)[::-1,0]
rnew = np.exp(rnew)

x=(x.T * rnew/r).T

#just some useful unit vectors...
#r2 = np.sum(x[:,:2]**2,axis=1)**0.5
#r = np.sum(x**2, axis=1)**0.5
#n_r = np.c_[x[:,0]/r2, x[:,1]/r2, np.zeros(len(x))]
#n_z = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.ones(N_gas)]
#n_phi = -np.cross(n_r, n_z)

mgas = np.repeat(M_gas/N_gas, N_gas)

Mr = M_BH + mgas.cumsum()

omega = S*(G*Mr/r**3)**0.5
v_K = omega * r

v = np.cross(np.c_[np.zeros_like(omega),np.zeros_like(omega),omega], x)

ugrav = G * np.sum(Mr/ r * mgas)

print ugrav,  (G*M_gas**2/R * 3/5)
if turbulence>0.0:
    vturb = TurbVelField(x, res=res_effective)
    vturb = (vturb.T * v_K).T
    mvturbSqr = 0.5*M_gas/N_gas*np.sum(vturb*vturb)
    v += vturb*np.sqrt(turbulence*ugrav/mvturbSqr)

if magnetic_field>0.0:
    B = TurbBField(x, res=res_effective)
    B = (B.T * v_K).T
    uB = np.sum(np.sum(B*B, axis=1) * 4*np.pi*R**3/3 /N_gas * 3.09e21**3)* 0.03979 *5.03e-54
    B = B * np.sqrt(magnetic_field*ugrav/uB)

v = v - np.average(v, axis=0)
x = x - np.average(x, axis=0)

print "Writing snapshot..."
F=h5py.File(filename, 'w')
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [N_gas,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["MassTable"] = [M_gas/N_gas,0,0,0,0, M_BH]
F["Header"].attrs["BoxSize"] = 0.0
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas)
F["PartType0"].create_dataset("Coordinates", data=x)
F["PartType0"].create_dataset("Velocities", data=v)
if magnetic_field > 0.0:
    F["PartType0"].create_dataset("MagneticField", data=B)
if M_BH > 0:
    F.create_group("PartType5")
    F["PartType5"].create_dataset("Masses", data=[M_BH,])
    F["PartType5"].create_dataset("Coordinates", data=[[0,0,0]])
    F["PartType5"].create_dataset("Velocities", data=[[0,0,0]])
F.close()
