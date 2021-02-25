#!/usr/bin/env python
"""                                                                            
MakeCloud: "Believe me, we've got some very turbulent clouds, the best clouds. You're gonna love it."

Usage: MakeCloud.py [options]

Options:                                                                       
   -h --help            Show this screen.
   --R=<pc>             Outer radius of the cloud in pc [default: 20.0]
   --M=<msun>           Mass of the cloud in msun [default: 1e5]
   --filename=<name>    Name of the IC file to be generated
   --N=<N>              Number of gas particles [default: 125000]
   --MBH=<msun>         Mass of the central black hole [default: 0.0]
   --central_star       Sets the central sink (MBH>0) to be a ZAMS star
   --central_SN         Sets the central sink (MBH>0) to go supernova
   --density_exponent=<f>   Power law exponent of the density profile [default: 0.0]
   --spin=<f>           Spin parameter: fraction of binding energy in solid-body rotation [default: 0.0]
   --turb_type=<s>      Type of initial turbulent velocity (and possibly density field): 'gaussian' or 'full' [default: gaussian]
   --turb_sol=<f>       Fraction of turbulence in solenoidal modes, used when turb_type is 'gaussian' [default: 0.5]
   --alpha_turb=<f>     Turbulent virial parameter (BM92 convention: 2Eturb/|Egrav|) [default: 2.]
   --bturb=<f>          Magnetic energy as a fraction of the binding energy [default: 0.01]
   --bfixed=<f>         Magnetic field in magnitude in code units, used instaed of bturb if not set to zero [default: 0]
   --minmode=<N>        Minimum populated turbulent wavenumber for Gaussian initial velocity field, in units of pi/R [default: 2]
   --turb_path=<name>   Path to store turbulent velocity fields so that we only need to generate them once [default: $HOME/turb]
   --glass_path=<name>  Contains the root path of the glass ic [default: $HOME/glass_orig.npy]
   --G=<f>              Gravitational constant in code units [default: 4300.71]
   --boxsize=<f>        Simulation box size
   --warmgas            Add warm ISM envelope in pressure equilibrium that fills the box with uniform density.
   --phimode=<f>        Relative amplitude of m=2 density perturbation (e.g. for Boss-Bodenheimer test) [default: 0.0]
   --localdir           Changes directory defaults assuming all files are used from local directory.
   --B_unit=<gauss>     Unit of magnetic field in gauss [default: 1e4]
   --length_unit=<pc>   Unit of length in pc [default: 1]
   --mass_unit=<msun>   Unit of mass in M_sun [default: 1]
   --v_unit=<m/s>       Unit of velocity in m/s [default: 1]
   --sinkbox=<f>        Setup for light seeds in a turbulent box problem - parameter is the maximum seed mass in solar [default: 0.0]
   --turb_seed=<N>      Random seed for turbulence initialization [default: 42]
   --param_only         Just makes the parameters file, not the IC
   --fixed_ncrit=<f>    Fixes ncrit to a specific value [default: 0.0]
   --makebox            Creates a second box IC of equivalent volume and mass to the cloud
"""
#Example:  python MakeCloud.py --M=1000 --N=1e7 --R=1.0 --localdir --warmgas --param_only


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

def TurbField(res=256, minmode=2, maxmode = 64, sol_weight=1., seed=42):
    freqs = fftpack.fftfreq(res)
    freq3d = np.array(np.meshgrid(freqs,freqs,freqs,indexing='ij'))
    intfreq = np.around(freq3d*res)
    kSqr = np.sum(np.abs(freq3d)**2,axis=0)
    intkSqr = np.sum(np.abs(intfreq)**2, axis=0)
    VK = []

    vSqr = 0.0
    # apply ~k^-2 exp(-k^2/kmax^2) filter to white noise to get x, y, and z components of velocity field
    for i in range(3):
        np.random.seed(seed+i)
        rand_phase = fftpack.fftn(np.random.normal(size=kSqr.shape)) # fourier transform of white noise
        vk = rand_phase * (float(minmode)/res)**2 / (kSqr+1e-300)
        #vk[intkSqr < minmode**2] = 0.0     # freeze out modes lower than minmode
#        print(intkSqr[intkSqr < minmode**2])
        vk[intkSqr==0] = 0.0
#        vk[intkSqr>0] *= np.exp(-minmode**2/intkSqr)
        vk[intkSqr < minmode**2] *= intkSqr[intkSqr < minmode**2]**2/minmode**4 # smoother filter than mode-freezing; should give less "ringing" artifacts
        vk *= np.exp(-intkSqr/maxmode**2)

        VK.append(vk)
    VK = np.array(VK)
    
    vk_new = np.zeros_like(VK)
    
    # do projection operator to get the correct mix of compressive and solenoidal
    for i in range(3):
        for j in range(3):
            if i==j:
                vk_new[i] += sol_weight * VK[j]
            vk_new[i] += (1 - 2 * sol_weight) * freq3d[i]*freq3d[j]/(kSqr+1e-300) * VK[j]
    vk_new[:,kSqr==0] = 0.0
    VK = vk_new
    
    vel = np.array([fftpack.ifftn(vk).real for vk in VK]) # transform back to real space
    vel -= np.average(vel,axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
    vel = vel / np.sqrt(np.sum(vel**2,axis=0).mean()) # normalize so that RMS is 1
    return np.array(vel)


arguments = docopt(__doc__)
R = float(arguments["--R"])
M_gas = float(arguments["--M"])
N_gas = int(float(arguments["--N"])+0.5)
M_BH = float(arguments["--MBH"])
central_star = arguments["--central_star"]
central_SN = arguments["--central_SN"]
spin = float(arguments["--spin"])
turbulence = float(arguments["--alpha_turb"])/2
turb_type = arguments["--turb_type"]
seed = int(float(arguments["--turb_seed"])+0.5)
turb_sol = float(arguments["--turb_sol"])
magnetic_field = float(arguments["--bturb"])
bfixed = float(arguments["--bfixed"])
minmode = int(arguments["--minmode"])
filename = arguments["--filename"]
turb_path = arguments["--turb_path"]
glass_path = arguments["--glass_path"]
G = float(arguments["--G"])
warmgas = arguments["--warmgas"]
localdir = arguments["--localdir"]
param_only = arguments["--param_only"]
B_unit = float(arguments["--B_unit"])
length_unit = float(arguments["--length_unit"])
mass_unit = float(arguments["--mass_unit"])
v_unit = float(arguments["--v_unit"])
sinkbox = float(arguments["--sinkbox"])
makebox = arguments["--makebox"]
fixed_ncrit=float(arguments["--fixed_ncrit"])
density_exponent=float(arguments["--density_exponent"])

if sinkbox:
    turb_type = 'full'
    
#B_unit = 1e4
#length_unit = 1.0
#mass_unit = 1.0
#v_unit = 1.0

if localdir:
    turb_path = "turb"
    glass_path = "glass_256.npy"

if arguments["--boxsize"] is not None:
#    print(arguments["--boxsize"])
    boxsize = float(arguments["--boxsize"])/length_unit
elif sinkbox:
    boxsize = 2*R
else:
    boxsize = 10*R

res_effective = int(N_gas**(1.0/3.0)+0.5)
phimode=float(arguments["--phimode"])

if filename is None:
    filename = "M%3.2g_"%(M_gas) + ("MBH%g_"%(M_BH) if M_BH>0 else "") + ("rho_exp%g_"%(-density_exponent) if density_exponent<0 else "") + "R%g_S%g_T%g_B%g_Res%d_n%d_sol%g"%(R,spin,turbulence,magnetic_field,res_effective,minmode,turb_sol) +  ("_%d"%seed) + ("_SN" if central_SN else "") + ".hdf5"
    filename = filename.replace("+","").replace('e0','e')
    filename = "".join(filename.split())
    
#    print "Cloud density: ", (np.sum(mgas)*1e10/mass_unit/(4.0/3.0*3.141*(R*1000/length_unit)**3)), " M_sun/pc^3", '   ',  (np.sum(mgas)*1e10/mass_unit/(4.0/3.0*3.141*(R*1000/length_unit)**3)/24532.3*1e6), " mu^(-1) cm^(-3)" 
    #n_crit mased on assumption that dm=M_jeans, meaning that densest is still resolved by NJ particles
    delta_m = M_gas/mass_unit/N_gas
    #rhocrit = 421/ delta_m**2
    rho_avg = 3*M_gas/(R)**3/(4*np.pi)
    softening = 3.11e-5 # ~6.5 AU, minimum sink radius is 2.8 times that (~18 AU)
    if fixed_ncrit:
        ncrit=fixed_ncrit
    else:
        #ncrit = 8920 / delta_m**2 #1.0e11
        ncrit = 9.45e9 * (delta_m/0.001)**(-2.0)
    tff = 8.275e-3 * rho_avg**-0.5
    L = (4*np.pi*R**3/3)**(1./3) /length_unit# volume-equivalent box size
    vrms = (6/5 * G * M_gas / R)**0.5  / v_unit * turbulence**0.5
    if turbulence:
        tcross = L/vrms
    else:
        tcross = tff

    turbenergy = 0.019111097819633344*vrms**3/L # ST_Energy sets the dissipation rate of SPECIFIC energy ~ v^2 / (L/v) ~ v^3/L
#    print(tff)
#   ncrit=(360684.5/((M_gas*1e10/mass_unit/N_gas)**2))
    #print "n_crit assuming NJ*dm=M_jeans: ", ncrit ,"T10^3 NJ(^-2) mu^(-4) cm^(-3)", np.log10(ncrit)
    #10^10 cm^-3 -> 2.45*10^8*mu*M_sun/pc^3, where mu is molecular weight
#    print "dx_min: ", ((np.sum(mgas)*1e10/mass_unit/(2.45e8*ncrit/1e10))**(1/3.0)), "T10^(-1) NJ(^2/3) mu^(4/3) pc"
    paramsfile = str(open(os.path.realpath(__file__).replace("MakeCloud.py","params.txt"), 'r').read())

    replacements = {"NAME": filename.replace(".hdf5",""), "DTSNAP": tff/150, "MAXTIMESTEP": tff/300, "SOFTENING": softening, "GASSOFT": 2.0e-8, "TMAX": tff*5, "RHOMAX": ncrit, "BOXSIZE": boxsize/length_unit, "OUTFOLDER": "output", "WIND_PART_MASS": delta_m/10.0, "BH_SEED_MASS": delta_m/2.0 , "TURBDECAY": tcross/2, "TURBENERGY": turbenergy, "TURBFREQ": tcross/20, "TURB_KMIN": int(100 * 2*np.pi/L)/100., "TURB_KMAX": int(100*4*np.pi/(L)+1)/100.}
    
    for k in replacements.keys():
        paramsfile = paramsfile.replace(k, str(replacements[k])) 
    open("params_"+filename.replace(".hdf5","")+".txt", "w").write(paramsfile)
    if makebox:
        replacements["NAME"] = filename.replace(".hdf5","_BOX")
        replacements["BOXSIZE"] = L
        paramsfile = str(open(os.path.realpath(__file__).replace("MakeCloud.py","params.txt"), 'r').read())
        for k in replacements.keys():
            paramsfile = paramsfile.replace(k, str(replacements[k]))         
        open("params_"+filename.replace(".hdf5","")+"_BOX.txt", "w").write(paramsfile)
if param_only:
    print('Parameters only run, exiting...')
    exit()

mgas = np.repeat(M_gas/N_gas, N_gas)

if turb_type=='full':
    ft = h5py.File("/panfs/ds08/hopkins/mgrudic/turb/supersonic128/snapshot_%s.hdf5"%(str(seed).zfill(3)), 'r')

#    ft = h5py.File("/panfs/ds08/hopkins/mgrudic/turb/nomhd/snapshot_002_c.hdf5")
    x = np.float64(np.array(ft["PartType0"]["Coordinates"]))-0.5
    Ns = len(x)

    r = np.sum(x**2,axis=1)**0.5
    if sinkbox:
        subset = np.ones_like(r, dtype=np.bool)
        N_gas = Ns
        mgas = np.repeat(M_gas/N_gas, N_gas)
        res_effective = int(N_gas**(1.0/3.0)+0.5)
    else:
        subset = r < 0.5
    if "MagneticField" in ft["PartType0"].keys():
        B = np.float64(np.array(ft["PartType0"]["MagneticField"]))[subset]
    else:
        B = 0*x

    v = np.float64(np.array(ft["PartType0"]["Velocities"]))[subset]
    h = np.float64(np.array(ft["PartType0"]["SmoothingLength"]))[subset]
    m = np.float64(np.array(ft["PartType0"]["Masses"]))[subset]

    x = x[subset]
    if sinkbox:
        xchoice = np.arange(len(x))
    else:
        xchoice = np.random.choice(np.arange(len(x)),size=N_gas,replace=False)

#    uB = np.sum(np.sum(B*B, axis=1) * 4*np.pi/3 *h**3 /32 * 3.09e21**3)* 0.03979 *5.03e-54
    plasma_beta = 0.5*np.sum(m*np.sum(v**2,axis=1))/uB

    x, v, B, h, m = x[xchoice], v[xchoice], B[xchoice], h[xchoice], m[xchoice]
    
    x = x*2*R
    h = h*2*R
    #B /= (0.5/(2*R))**1.5
    r = np.sum(x**2,axis=1)**0.5
else:
    x = np.load(glass_path)
    Nx = len(x)

    while len(x)*np.pi*4/3 / 8 < N_gas:
        print("Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"%(N_gas * 8 /(4*np.pi/3), len(x)))
        x = np.concatenate([x/2 + i * np.array([0.5,0,0]) + j * np.array([0,0.5,0]) + k * np.array([0, 0, 0.5]) for i in range(2) for j in range(2) for k in range(2)])
        Nx = len(x)
    print("Glass loaded!")
    x = 2*(x-0.5)
    print("Computing radii...")
    r = cdist(x, [np.zeros(3)])[:,0] #np.sum(x**2, axis=1)**0.5
    print("Done! Sorting coordinates...")
    x = x[r.argsort()][:N_gas]
    print("Done! Rescaling...")
    x *= (float(Nx) / N_gas * 4*np.pi/3 / 8)**(1./3)*R
    print("Done! Recomupting radii...")
    r = cdist(x, [np.zeros(3)])[:,0]
#    r = np.sum(x**2,axis=1)**0.5
    x, r = x/r.max(), r/r.max()
#    rnew = r * R
    print("Doing density profile...")
    if density_exponent<0:
        rho_form = lambda r: (r+R/1000)**(density_exponent) #change this function to get a different radial density profile; normalization does not matter as long as rmin and rmax are properly specified
        #rho_form = lambda r: (r+R/1000)**-1.5
        rmin = 0.
        rho_norm = quad(lambda r: rho_form(r) * 4 * np.pi * r**2, rmin, R)[0]
        rho = lambda r: rho_form(r) / rho_norm
        rnew = odeint(lambda rphys, r3: np.exp(r3)/(4*np.pi*np.exp(rphys)**3*rho(np.exp(rphys))), np.log(R), np.log(r[:-1]**3), atol=1e-12, rtol=1e-12)[::-1,0]
        rnew = np.exp(rnew)
    else:
        print("Using constant density")
        #rho_form = lambda r: 1. #constant density
        rnew = r *R
    x=(x.T * rnew/r).T
    r = np.sum(x**2, axis=1)**0.5
    x, r = x[r.argsort()], r[r.argsort()]

    if turb_type=='gaussian':
#        if turb_path is not 'none': # this is for saving turbulent fields we have already generated
        if not os.path.exists(turb_path): os.makedirs(turb_path)
        fname = turb_path + "/vturb%d_sol%g_seed%d.npy"%(minmode,turb_sol, seed)
        if not os.path.isfile(fname):
            vt = TurbField(minmode=minmode, sol_weight=turb_sol, seed=seed)

            nmin, nmax = vt.shape[-1]// 4, 3*vt.shape[-1]//4

            vt = vt[:,nmin:nmax, nmin:nmax, nmin:nmax]  # we take the central cube of size L/2 so that opposide sides of the cloud are not correlated
            np.save(fname, vt)
        else:
            vt = np.load(fname)
        
        xgrid = np.linspace(-R,R,vt.shape[-1])
        v = []
        for i in range(3):
            v.append(interpolate.interpn((xgrid,xgrid,xgrid),vt[i,:,:,:],x))
        v = np.array(v).T
    print("Coordinates obtained!")
#x += np.random.normal(size=(N_gas,3))*R/20
        
Mr = M_BH + mgas.cumsum()
if sinkbox:
    ugrav= G * M_gas**2 / R
else:
    ugrav = G * np.sum(Mr/ r * mgas)
print(ugrav)
E_rot = spin * ugrav
I_z = np.sum(mgas * (x[:,0]**2+x[:,1]**2))
omega = (2*E_rot/I_z)**0.5
#omega = spin * np.sqrt(G*(M_BH+M_gas)/R**3)

v -= np.average(v,axis=0)
Eturb = 0.5*M_gas/N_gas*np.sum(v**2)
v *= np.sqrt(turbulence*ugrav/Eturb)

v += np.cross(np.c_[np.zeros_like(omega),np.zeros_like(omega),omega], x)


if ( (magnetic_field>0.0) or (bfixed>0) ) and turb_type != 'full':
    B = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.ones(N_gas)]
    uB = np.sum(np.sum(B*B, axis=1) * (R*length_unit)**3 /N_gas) * 2.463e17 
    if (bfixed>0):
        B = B * bfixed
    else:
        B = B * np.sqrt(magnetic_field*ugrav/uB) # in gauss


v = v - np.average(v, axis=0)
x = x - np.average(x, axis=0)

r, phi = np.sum(x**2,axis=1)**0.5, np.arctan2(x[:,1],x[:,0])
theta = np.arccos(x[:,2]/r)
phi += phimode*np.sin(2*phi)/2
x = r[:,np.newaxis]*np.c_[np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)]


#print(h)
if turb_type=='full':
    if not sinkbox:
        import meshoid
        M = meshoid.meshoid(x,mgas)
        rho, h = M.Density(), M.SmoothingLength()
    else:
        rho = (32*mgas/(4*np.pi/3 * h**3))
#    uB = np.sum(np.sum(B*B, axis=1) * 4*np.pi/3 *h**3 /32 * 3.09e21**3)* 0.03979 *5.03e-46
    beta = np.sum(0.5*mgas*np.sum(v**2,axis=1))/uB
    B *= np.sqrt(beta/plasma_beta)
#    uB = np.sum(np.sum(B*B, axis=1) * 4*np.pi/3 *h**3 /32 * 3.09e21**3)* 0.03979 *5.03e-54
    
#print(x.mean(axis=0))
u = np.ones_like(mgas)*0.101/2.0 #/2 needed because it is molecular
if warmgas:
    # assuming 10K vs 10^4K gas: factor of ~10^3 density contrast
    rho_warm = M_gas*3/(4*np.pi*R**3) / 1000
    M_warm = (boxsize**3 - (4*np.pi*R**3 / 3)) * rho_warm # mass of diffuse box-filling medium
    N_warm = int(M_warm/(M_gas/N_gas))
#    print(N_warm)
    x_warm = boxsize*np.random.rand(N_warm, 3) - boxsize/2
    x_warm = x_warm[np.sum(x_warm**2,axis=1) > R**2]
    N_warm = len(x_warm)
    x = np.concatenate([x, x_warm])
    v = np.concatenate([v, np.zeros((N_warm,3))])
    Bmag = np.average(np.sum(B**2,axis=1))**0.5
    B = np.concatenate([B, np.repeat(Bmag,N_warm)[:,np.newaxis] * np.array([0,0,1])])
    mgas = np.concatenate([mgas, np.repeat(M_gas/N_gas,N_warm)])
    u = np.concatenate([u, np.repeat(101.,N_warm)])
    # old kludgy way of setting up diffuse medium with uniform B field; probably don't use this
    # N_warm = int(warmgas*N_gas+0.5)
    # sigma_warm = 2*R*10*warmgas**(1./3)
    # x_warm = np.random.normal(size=(N_warm,3))*sigma_warm
    # r_warm = np.sum(x_warm**2,axis=1)**0.5
    # R_warm = np.sum(x_warm[:,:2]**2,axis=1)**0.5
    # x = np.concatenate([x, x_warm])
    # v = np.concatenate([v, np.zeros((N_warm,3))])
    # Bmag = np.average(np.sum(B**2,axis=1))**0.5
    # B = np.concatenate([B, Bmag * np.exp(-R_warm**2/(2*sigma_warm**2))[:,np.newaxis] * np.array([0,0,1])])
    # mgas = np.concatenate([mgas, np.repeat(M_gas/N_gas,N_warm)])

else:
    N_warm = 0


if turb_type!='full': 
    rho = np.repeat(3*M_gas/(4*np.pi*R**3), len(mgas))
    if warmgas: rho[-N_warm:] /= 1000
    h = (32*mgas/rho)**(1./3)

#if (arguments["--boxsize"] is not None or arguments["--warmgas"]) or sinkbox:

x += boxsize/2 # cloud is always centered at (boxsize/2,boxsize/2,boxsize/2)

if sinkbox:
    m_min = M_gas/N_gas * 10
    m_max = sinkbox
    if sinkbox < m_min: print("Maximum sink mass is less than the minimum of 10 times the gas mass resolution. Increase resolution or implement a different sampling scheme.")

    N_sinks = int(round(m_max/m_min))

    m_sinks = (1./m_min - (1/m_min - 1/m_max) * np.random.rand(N_sinks))**-1. # randomly sample from M^-2 mass function between M_max and M_min
    x_sinks = np.random.rand(N_sinks,3)*boxsize # random coordinates
    v_sinks = np.random.normal(size=(N_sinks,3)) * (np.sum(v**2,axis=1).mean() / 3)**0.5 # random velocities equal to RMS gas velocity        



dx= x/length_unit - np.array([1.,1.,1.])*boxsize/2/length_unit

print("Writing snapshot...")

F=h5py.File(filename, 'w')
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [N_gas+N_warm,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["NumPart_Total"] = [N_gas+N_warm,0,0,0,0,(1 if M_BH>0 else 0)]
F["Header"].attrs["MassTable"] = [M_gas/N_gas/mass_unit,0,0,0,0, M_BH/mass_unit]
F["Header"].attrs["BoxSize"] = boxsize/length_unit
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas/mass_unit)
F["PartType0"].create_dataset("Coordinates", data=x/length_unit)
F["PartType0"].create_dataset("Velocities", data=v/v_unit)
F["PartType0"].create_dataset("ParticleIDs", data=1+np.arange(N_gas+N_warm)+(1 if M_BH>0 else 0))
F["PartType0"].create_dataset("InternalEnergy", data=u*(1/v_unit)**2)
F["PartType0"].create_dataset("Density", data=rho/mass_unit/(1/length_unit)**3)
F["PartType0"].create_dataset("SmoothingLength", data=h*1/length_unit)
if M_BH>0:
    F.create_group("PartType5")
    #Let's add the sink at the center
    F["PartType5"].create_dataset("Masses", data=np.array([M_BH/mass_unit]))
    F["PartType5"].create_dataset("Coordinates", data=np.array([1.,1.,1.])*boxsize/2/length_unit) #at the center
    F["PartType5"].create_dataset("Velocities", data=np.array([0.,0.,0.])) #at rest
    F["PartType5"].create_dataset("ParticleIDs", data=np.array([1]))
    F["PartType5"].create_dataset("SmoothingLength", data=np.array([np.mean(h/length_unit)])/2.)
    #Advanced properties for sinks
    F["PartType5"].create_dataset("BH_Mass", data=M_BH/mass_unit) #all the mass in the sink/protostar/star
    F["PartType5"].create_dataset("BH_Mass_AlphaDisk", data=np.array([0.])) #starts with no disk
    F["PartType5"].create_dataset("BH_Mdot", data=np.array([0.])) #starts with no mdot
    F["PartType5"].create_dataset("BH_Specific_AngMom", data=np.array([0.])) #starts with no angular momentum
    F["PartType5"].create_dataset("SinkRadius", data=np.array([softening])) #Sinkradius set to softening
    F["PartType5"].create_dataset("StellarFormationTime", data=np.array([0.]))
    #Stellar properties
    if (central_star or central_SN):
        if central_star:
            print("Assuming central sink is a ZAMS star")
            F["PartType5"].create_dataset("ProtoStellarStage", data=np.array([5],dtype=np.int32), dtype=np.int32) #starts as ZAMS star
        else:
            print("Assuming central sink is a ZAMS star about to go supernova")
            F["PartType5"].create_dataset("ProtoStellarStage", data=np.array([6],dtype=np.int32), dtype=np.int32) #starts as ZAMS star going SN
        #Set guess for ZAMS stellar radius, will be overwritten
        if ((M_BH)>1.0):
            R_ZAMS = (M_BH)**0.57
        else:
            R_ZAMS = (M_BH)**0.8
        F["PartType5"].create_dataset("ProtoStellarRadius_inSolar", data=np.array([R_ZAMS])) #Sinkradius set to softening
        F["PartType5"].create_dataset("StarLuminosity_Solar", data=np.array([0.])) #dummy
        F["PartType5"].create_dataset("Mass_D", data=np.array([0.])) #No D left
    else:
        print("Assuming central sink is a pre-collapse object")
        F["PartType5"].create_dataset("ProtoStellarStage", data=np.array([0],dtype=np.int), dtype=np.int32) #starts as pre-collapse
        F["PartType5"].create_dataset("ProtoStellarRadius_inSolar", data=np.array([2.])) #dummy value
        F["PartType5"].create_dataset("StarLuminosity_Solar", data=np.array([0.])) #dummy
        F["PartType5"].create_dataset("Mass_D", data=np.array([M_BH/mass_unit])) #100% of D left

    
if magnetic_field > 0.0:
    F["PartType0"].create_dataset("MagneticField", data=B/B_unit)
if sinkbox:
    F.create_group("PartType5")
    F["PartType5"].create_dataset("Masses", data=m_sinks)
    F["PartType5"].create_dataset("Coordinates", data=x_sinks)
    F["PartType5"].create_dataset("Velocities", data=v_sinks)
    F["PartType5"].create_dataset("ParticleIDs", data=np.arange(N_gas+N_warm, N_gas+N_warm+N_sinks))
F.close()

if makebox:
    F=h5py.File(filename.replace(".hdf5","_BOX.hdf5"), 'w')
    F.create_group("PartType0")
    F.create_group("Header")
    F["Header"].attrs["NumPart_ThisFile"] = [N_gas,0,0,0,0,0]
    F["Header"].attrs["NumPart_Total"] = [N_gas,0,0,0,0,0]
    F["Header"].attrs["MassTable"] = [M_gas/N_gas/mass_unit,0,0,0,0,0]
    F["Header"].attrs["BoxSize"] = (4 * np.pi * R**3 / 3)**(1./3) / length_unit
    F["Header"].attrs["Time"] = 0.0
    F["PartType0"].create_dataset("Masses", data=mgas[:N_gas]/mass_unit)
    F["PartType0"].create_dataset("Coordinates", data = F["Header"].attrs["BoxSize"] * np.random.rand(N_gas,3))
    F["PartType0"].create_dataset("Velocities", data=np.zeros((N_gas,3)))
    F["PartType0"].create_dataset("ParticleIDs", data=1+np.arange(N_gas))
    if magnetic_field > 0.0:
        F["PartType0"].create_dataset("MagneticField", data=B[:N_gas]/B_unit)
    F.close()
