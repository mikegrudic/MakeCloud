#!/usr/bin/env python
"""                                                                            
MakeCloud: "Believe me, we've got some very turbulent clouds, the best clouds. You're gonna love it."
This is an alternate version that creates a homogeneous spherical medium around a sink particle, meant to do a Bondi-accretion test. By default it uses default GIZMO units

Usage: MakeCloud_Bondi.py [options]

Options:                                                                       
   -h --help                Show this screen.
   --R_over_Rsonic=<f>      Outer radius of the cloud in Rsonic [default: 16.0]
   --Rs_over_Rsonic=<f>     Ratio of sink radius to sonic radius [default: 1.0]
   --rho_gas=<msun>         Density of the gas around the sink at inifinity, in msolar/pc^3 [default: 0.01]
   --filename=<name>        Name of the IC file to be generated
   --N=<N>                  Number of gas particles [default: 583200]
   --MBH=<msun>             Mass of the sink at center of the sphere, should be much larger than the gas mass, in msolar [default: 1.0]
   --boxsize=<f>            Simulation box size in pc
   --length_unit=<pc>       Unit of length in pc [default: 1000]
   --mass_unit=<msun>       Unit of mass in M_sun [default: 1e10]
   --v_unit=<m/s>           Unit of velocity in m/s [default: 1000]
   --GMC_units              Sets units appropriate for GMCs, so pc, m/s, m_sun, tesla
   --localdir               Changes directory defaults assuming all files are used from local directory.
   --R_cut_out=<f>          Distance around the sink to be left empty in pc [default: 0.0001]
"""

from __future__ import print_function
import numpy as np
from scipy import fftpack, interpolate, ndimage
from scipy.integrate import quad, odeint, solve_bvp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import sys
import h5py
import os
from docopt import docopt
#from pykdgrav import Potential
arguments = docopt(__doc__)
GMC_units = arguments["--GMC_units"]
length_unit = float(arguments["--length_unit"])
mass_unit = float(arguments["--mass_unit"])
v_unit = float(arguments["--v_unit"])
if GMC_units:
    length_unit = 1.0
    mass_unit = 1.0
    v_unit = 1.0
    

R_over_Rsonic = float(arguments["--R_over_Rsonic"])
Rs_over_Rsonic = float(arguments["--Rs_over_Rsonic"])
rho_gas = float(arguments["--rho_gas"])/(mass_unit/(length_unit**3.0)) #msolar/pc^3 to code units
N_gas = int(float(arguments["--N"])+0.5)
M_BH = float(arguments["--MBH"])/mass_unit #mstar to code units
R_cut_out = float(arguments["--R_cut_out"])/length_unit
filename = arguments["--filename"]
localdir = arguments["--localdir"]



#set gravitational constant
G = 4325.69 /length_unit / (v_unit**2) * (mass_unit)
    
if localdir:
    turb_path = "turb"
    glass_path = "glass_256.npy"

#get rsink radius from sonic radius
csound=200/v_unit #200 m/s
rsonic_Bondi = G*M_BH/2.0/(csound**2)/length_unit
msonic=4.0*np.pi/3.0*rho_gas*(rsonic_Bondi**3.0)
Rsink = Rs_over_Rsonic*rsonic_Bondi
R = R_over_Rsonic*rsonic_Bondi
print( "Radius set as %g pc"%(R*length_unit))
if arguments["--boxsize"] is not None:
#    print((arguments["--boxsize"]))
    boxsize = float(arguments["--boxsize"])*length_unit
else:
    boxsize = 10*R
center = np.ones(3)*boxsize/2.0
res_effective = int(N_gas**(1.0/3.0)+0.5)

#Load Bondi Spherical solution
IC_data=np.loadtxt("BONDI_SOLUTION.dat") #load IC data from Hubber
data_r=IC_data[:,0]*rsonic_Bondi #coordinate originally in Rsonic
data_within_R_index=(data_r<=R);data_r=data_r[data_within_R_index] #restrict to within R radius
data_mass_in_r=IC_data[data_within_R_index,1]*msonic#gas mass in units of 4*pi*rho0*Rsonic^3/3
data_density=IC_data[data_within_R_index,2]*rho_gas #density originally in rho0
data_vr=IC_data[data_within_R_index,3]*csound #velocity originally in csound
M_gas=np.max(data_mass_in_r) #normalize to total mass
#Get glass
mgas = np.repeat(M_gas/N_gas, N_gas) #gas mass init
x = 2*(np.load(glass_path)-0.5) #load glass to have basic structure
Nx=len(x[:,0]); #original glass size
r = np.sum(x**2,axis=1)**0.5 #calculate radius
x = x[r.argsort()][:N_gas] #sort the particles by radius, it should be spherically symmetric, now take the right number of particles
x *= (float(Nx) / N_gas * 4*np.pi/3 / 8)**(1./3) #scale up the sphere to have unit radius
r = np.sum(x**2,axis=1)**0.5 #recalculate radius
#Strech the particles based on how much mass is in a given radius
int_mass=np.cumsum(mgas) #integrated mass
newr=np.interp(int_mass, data_mass_in_r, data_r) #calculate the radius the particles should be at to have the sane mass within the radius a sthe solution
stretch=newr/r #stretch factor
x[:,0] *= stretch;x[:,1] *= stretch;x[:,2] *= stretch;
u = np.ones_like(mgas)*0.101*((1000/v_unit)**2)/2.0 #/2 needed because it is molecular
rho = np.interp(int_mass, data_mass_in_r, data_density) #interpolate the density
h = (32*mgas/rho)**(1./3)
vr = np.interp(int_mass, data_mass_in_r, data_vr) #interpolate the velocity
v = -x; v[:,0] *= vr/newr; v[:,1] *= vr/newr; v[:,2] *= vr/newr;
#print chek data
print( 'mdot avg: %g std %g'%(np.mean(data_density*data_r*data_r*data_vr*np.pi*4.0),np.std(data_density*data_r*data_r*data_vr*np.pi*4.0)))
print( 'density')
print( data_density)
print( 'density_alt')
altdens=np.diff(data_mass_in_r)/np.diff(data_r)/(4.0*np.pi*data_r[1:]*data_r[1:])
print( altdens)
print( 'relative')
print( data_density[1:]/altdens)
#Calculate NEWSNK parameters
print( 'Calculating t_radial...')
for z in [0.125,0.5,1.0,2.0,4.0]:
    print( '\t At %g Rsonic = %g pc'%(z,(rsonic_Bondi*z*length_unit)))
    ind=(newr<=(z*rsonic_Bondi))
    print( '\t mass enclosed %g in msun'%(np.sum(mgas[ind])*mass_unit))
    #tot_weight=np.sum(mgas[ind]/rho[ind])
    tot_weight=4.0/3.0*np.pi*np.max(newr[ind])**3
    x_dot_v=np.sum(x*v,axis=1)
    t_rad=-np.sum(mgas[ind])*tot_weight/np.sum(4.0*np.pi*(x_dot_v*newr*mgas)[ind])
    print( '\t t_rad=%g in code units'%(t_rad))
    print( '\t m/t_rad=%g in code units'%(np.sum(mgas[ind])/t_rad))
    #from IC
    ind=np.arange(len(data_r))[(data_r<=(z*rsonic_Bondi))]
    ind2=np.argmax(data_r[ind])
    dm_ic=np.diff(data_mass_in_r)
    t_rad_IC=(data_mass_in_r[ind2]*(4.0/3.0*np.pi)*data_r[ind2]**3)/(4.0*np.pi*np.sum(data_r[ind]*data_r[ind]*data_vr[ind]*dm_ic[ind]))
    print( '\t t_rad_IC=%g in code units'%(t_rad_IC))
    print( '\t m_IC/t_rad_IC=%g in code units'%(data_mass_in_r[ind2]/t_rad_IC))
#Calculate flux
print( 'Calculating density ...')
dr=np.diff(newr)
rho_alt=(mgas[1:]/dr)/(4.0*np.pi*(newr[1:]**2))
print( rho)
print( rho_alt)
print( rho[1:]/rho_alt)
#keep the one sthat are not too close
ind=newr>R_cut_out
print("Removing %d particles that are inside R_cut_out of %g"%((N_gas-np.sum(ind)),R_cut_out))
N_gas=np.sum(ind)
M_gas=np.sum(mgas[ind])

NGBvals=[1,2,5,10,20,50,100,200,400,800,1600]
for ngb in NGBvals:
    print("Radius at Ngbfactor of %g is %g pc"%(ngb,newr[32*ngb]))

#center coordinates
x = x + boxsize/2.0
print("Writing snapshot...")

if filename is None:
    filename = "rho%3.2g_"%(rho_gas*(mass_unit/(length_unit**3))) + ("MBH%g_"%(mass_unit*M_BH) if M_BH>0 else "") + "R_over_Rsonic%g_Res%d_Rs_over_Rsonic%g"%(R_over_Rsonic,res_effective,Rs_over_Rsonic) + ".hdf5"
    filename = filename.replace("+","").replace('e0','e')
    filename = "".join(filename.split())
    
F=h5py.File(filename, 'w')
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [N_gas,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["NumPart_Total"] = [N_gas,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["MassTable"] = [M_gas/N_gas,0,0,0,0, M_BH]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas[ind])
F["PartType0"].create_dataset("Coordinates", data=x[ind,:])
F["PartType0"].create_dataset("Velocities", data=v[ind,:])
F["PartType0"].create_dataset("ParticleIDs", data=np.arange(N_gas)+1)
F["PartType0"].create_dataset("InternalEnergy", data=u[ind])
F["PartType0"].create_dataset("Density", data=rho[ind])
F["PartType0"].create_dataset("SmoothingLength", data=h[ind])
if M_BH>0:
    F.create_group("PartType5")
    F["PartType5"].create_dataset("Masses", data=np.array([M_BH]))
    F["PartType5"].create_dataset("BH_Mass", data=np.array([M_BH]))
    F["PartType5"].create_dataset("Coordinates", data=center)
    F["PartType5"].create_dataset("Velocities", data=v[0,:]*0.0)
    F["PartType5"].create_dataset("ParticleIDs", data=np.array([N_gas+1]))
    F["PartType5"].create_dataset("SinkRadius", data=np.array([Rsink]))

F.close()

if GMC_units: 
    delta_m = M_gas/N_gas
    rhocrit = 421/ delta_m**2
    rho_avg = 3*M_gas/(R**3)/(4*np.pi)
    softening = 0.000173148 # 100AU/2.8 #(delta_m/rhocrit)**(1./3)
    ncrit = 1.0e11 #8920 / delta_m**2
    tff = 8.275e-3 * rho_avg**-0.5
    mdot_Bondi = np.exp(1.5)*np.pi*(G**2)*(M_BH**2)*rho_gas/(csound**3)
    tend_Bondi = 2.0*(2.0*G*M_BH/(csound**3))
    print( 'Bondi accretion parameters: \n \t Mgas:\t\t', M_gas, '\n \t Mdot:\t\t',mdot_Bondi, '\n \t Rsonic:\t',rsonic_Bondi,'\n \t Rsink:\t\t',Rsink, '\n \t t_end:\t\t',tend_Bondi, '\n \t m_acc:\t\t',tend_Bondi*mdot_Bondi, '\n \t f_acc:\t\t',tend_Bondi*mdot_Bondi)
    paramsfile = str(open(os.path.realpath(__file__).replace("MakeCloud_Bondi.py","params.txt"), 'r').read())

    replacements = {"NAME": "../ICs/"+filename.replace(".hdf5",""), "DTSNAP": tend_Bondi/200, "SOFTENING": softening, "GASSOFT": 2.0e-8, "TMAX": tend_Bondi, "RHOMAX": ncrit, "BOXSIZE": boxsize, "OUTFOLDER": "output_%g"%(Rs_over_Rsonic)}

    print(replacements["NAME"])
#    print(paramsfile)
    for k in replacements.keys():
        paramsfile = paramsfile.replace(k, str(replacements[k])) 
    open("params_"+filename.replace(".hdf5","")+".txt", "w").write(paramsfile)
    

