#!/usr/bin/env python
"""                                                                            
MakeCloud: "Believe me, we've got some very turbulent clouds, the best clouds. You're gonna love it."

Usage: MakeCloud.py [options]

Options:                                                                       
   -h --help            Show this screen.
   --R=<pc>             Outer radius of the cloud in pc [default: 10.0]
   --M=<msun>           Mass of the cloud in msun [default: 2e4]
   --filename=<name>    Name of the IC file to be generated
   --N=<N>              Number of gas particles [default: 2000000]
   --density_exponent=<f>   Power law exponent of the density profile [default: 0.0]
   --spin=<f>           Spin parameter: fraction of binding energy in solid-body rotation [default: 0.0]
   --omega_exponent=<f>  Powerlaw exponent of rotational frequency as a function of cylindrical radius [default: 0.0]
   --turb_slope=<f>      Slope of the turbulent power spectra [default: 2.0]
   --turb_sol=<f>       Fraction of turbulence in solenoidal modes [default: 0.5]
   --alpha_turb=<f>     Turbulent virial parameter (BM92 convention: 2Eturb/|Egrav|) [default: 2.]
   --bturb=<f>          Magnetic energy as a fraction of the binding energy [default: 0.1]
   --bfixed=<f>         Magnetic field in magnitude in code units, used instead of bturb if not set to zero [default: 0]
   --minmode=<N>        Minimum populated turbulent wavenumber for Gaussian initial velocity field, in units of pi/R [default: 2]
   --turb_path=<name>   Path to store turbulent velocity fields so that we only need to generate them once (defaults to ~/turb)
   --glass_path=<name>  Contains the the path of the glass file (defaults to your home directory)
   --boxsize=<f>        Simulation box size
   --Mstar=<msun>       Mass of the star/black hole, if any [default: 0.0]
   --v_star=<vx,vy,vz>  Velocity of the star [default: 0.0,0.0,0.0]
   --x_star=<x,y,z>     Position of the star, defaults to center of the box
   --star_stage=<N>     Evolutionary stage of the star/black hole [default: 7]
   --derefinement       Apply radial derefinement to ambient cells outside of 3* cloud radius
   --no_diffuse_gas     Remove diffuse ISM envelope fills the rest of the box with uniform density. 
   --phimode=<f>        Relative amplitude of m=2 density perturbation (e.g. for Boss-Bodenheimer test) [default: 0.0]
   --localdir           Changes directory defaults assuming all files are used from local directory.
   --B_unit=<gauss>     Unit of magnetic field in gauss [default: 1e4]
   --length_unit=<pc>   Unit of length in pc [default: 1]
   --mass_unit=<msun>   Unit of mass in M_sun [default: 1]
   --v_unit=<m/s>       Unit of velocity in m/s [default: 1]
   --turb_seed=<N>      Random seed for turbulence initialization [default: 42]
   --tmax=<N>           Maximum time to run the simulation to, in units of the freefall time [default: 5]
   --nsnap=<N>          Number of snapshots per freefall time [default: 150]
   --param_only         Just makes the parameters file, not the IC
   --fixed_ncrit=<f>    Fixes ncrit to a specific value [default: 0.0]
   --makebox            Creates a second box IC of equivalent volume and mass to the cloud
   --impact_dist=<b>    Initial separation between cloud centers of mass in units of the cloud radius  (0 is no cloud-cloud collision) [default: 0.0]
   --impact_param=<b>   Impact parameter of cloud-cloud collision in units of the cloud radius [default: 0.0]
   --v_impact=<v>       Impact velocity, in units of the cloud's RMS turbulent velocity [default: 1.0]
   --impact_axis=<x>    Axis along which collision occurs (z is along magnetic field lines) [default: x]
   --makecylinder       Creates a third, cylindrical IC of equivalent volume and mass to the cloud
   --cyl_aspect_ratio=<f>   Sets the aspect ratio of the cylinder, i.e. Length/Diameter [default: 10]
   --Z=<solar>          Metallicity of the cloud in Solar units (just for params file) [default: 1.0]
   --ISRF=<solar>       Interstellar radiation background of the cloud in Solar neighborhood units (just for params file) [default: 1.0]
"""
# Example:  python MakeCloud.py --M=1000 --N=1e7 --R=1.0 --localdir --param_only

import os
import numpy as np
from scipy import fftpack, interpolate
from scipy.spatial.distance import cdist
import h5py
from docopt import docopt


def get_glass_coords(N_gas, glass_path):
    x = np.load(glass_path)
    Nx = len(x)

    while len(x) * np.pi * 4 / 3 / 8 < N_gas:
        print(
            "Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"
            % (N_gas * 8 / (4 * np.pi / 3), len(x))
        )
        x = np.concatenate(
            [
                x / 2
                + i * np.array([0.5, 0, 0])
                + j * np.array([0, 0.5, 0])
                + k * np.array([0, 0, 0.5])
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )
        Nx = len(x)
    print("Glass loaded!")
    return x


def TurbField(res=256, minmode=2, maxmode=64, slope=2.0, sol_weight=1.0, seed=42):
    freqs = fftpack.fftfreq(res)
    freq3d = np.array(np.meshgrid(freqs, freqs, freqs, indexing="ij"))
    intfreq = np.around(freq3d * res)
    kSqr = np.sum(np.abs(freq3d) ** 2, axis=0)
    intkSqr = np.sum(np.abs(intfreq) ** 2, axis=0)
    VK = []

    # apply ~k^-2 exp(-k^2/kmax^2) filter to white noise to get x, y, and z components of velocity field
    for i in range(3):
        np.random.seed(seed + i)
        rand_phase = fftpack.fftn(
            np.random.normal(size=kSqr.shape)
        )  # fourier transform of white noise
        vk = rand_phase * (float(minmode) / res) ** 2 / (np.power(kSqr, slope/2.0) + 1e-300)
        vk[intkSqr == 0] = 0.0
        vk[intkSqr < minmode**2] *= (
            intkSqr[intkSqr < minmode**2] ** 2 / minmode**4
        )  # smoother filter than mode-freezing; should give less "ringing" artifacts
        vk *= np.exp(-intkSqr / maxmode**2)

        VK.append(vk)
    VK = np.array(VK)

    vk_new = np.zeros_like(VK)

    # do projection operator to get the correct mix of compressive and solenoidal
    for i in range(3):
        for j in range(3):
            if i == j:
                vk_new[i] += sol_weight * VK[j]
            vk_new[i] += (
                (1 - 2 * sol_weight)
                * freq3d[i]
                * freq3d[j]
                / (kSqr + 1e-300)
                * VK[j]
            )
    vk_new[:, kSqr == 0] = 0.0
    VK = vk_new

    vel = np.array(
        [fftpack.ifftn(vk).real for vk in VK]
    )  # transform back to real space
    vel -= np.average(vel, axis=(1, 2, 3))[
        :, np.newaxis, np.newaxis, np.newaxis
    ]
    vel = vel / np.sqrt(
        np.sum(vel**2, axis=0).mean()
    )  # normalize so that RMS is 1
    return np.array(vel)


arguments = docopt(__doc__)
R = float(arguments["--R"])
M_gas = float(arguments["--M"])
N_gas = int(float(arguments["--N"]) + 0.5)
M_star = float(arguments["--Mstar"])
v_star = np.array([float(v) for v in arguments["--v_star"].split(",")])
spin = float(arguments["--spin"])
omega_exponent = float(arguments["--omega_exponent"])
turbulence = float(arguments["--alpha_turb"]) / 2
seed = int(float(arguments["--turb_seed"]) + 0.5)
tmax = int(float(arguments["--tmax"]))
nsnap = int(float(arguments["--nsnap"]))
turb_slope = float(arguments["--turb_slope"])
turb_sol = float(arguments["--turb_sol"])
magnetic_field = float(arguments["--bturb"])
bfixed = float(arguments["--bfixed"])
minmode = int(arguments["--minmode"])
filename = arguments["--filename"]
diffuse_gas = not arguments["--no_diffuse_gas"]
localdir = arguments["--localdir"]
param_only = arguments["--param_only"]
B_unit = float(arguments["--B_unit"])
length_unit = float(arguments["--length_unit"])
mass_unit = float(arguments["--mass_unit"])
v_unit = float(arguments["--v_unit"])
t_unit = length_unit / v_unit
G = 4300.71 * v_unit**-2 * mass_unit / length_unit
makebox = arguments["--makebox"]
impact_param = float(arguments["--impact_param"])
impact_dist = float(arguments["--impact_dist"])
v_impact = float(arguments["--v_impact"])
impact_axis = arguments["--impact_axis"]
makecylinder = arguments["--makecylinder"]
cyl_aspect_ratio = float(arguments["--cyl_aspect_ratio"])
fixed_ncrit = float(arguments["--fixed_ncrit"])
density_exponent = float(arguments["--density_exponent"])
metallicity = float(arguments["--Z"])
ISRF = float(arguments["--ISRF"])
if arguments["--turb_path"]:
    turb_path = arguments["--turb_path"]
else:
    turb_path = os.path.expanduser("~") + "/turb"
if arguments["--glass_path"]:
    glass_path = arguments["--glass_path"]
else:
    glass_path = os.path.expanduser("~") + "/glass_orig.npy"
    if not os.path.exists(glass_path):
        import urllib.request

        print("Downloading glass file...")
        urllib.request.urlretrieve(
            "http://www.tapir.caltech.edu/~mgrudich/glass_orig.npy",
            glass_path,
            #            "https://data.obs.carnegiescience.edu/starforge/glass_orig.npy", glass_path
        )

if localdir:
    turb_path = "turb"
    glass_path = "glass_256.npy"

if arguments["--boxsize"] is not None:
    boxsize = float(arguments["--boxsize"])
else:
    boxsize = 10 * R

if arguments["--x_star"]:
    x_star = np.array([float(x) for x in arguments["--x_star"].split(",")])
else:  # default to center of box
    x_star = np.repeat(0.5 * boxsize, 3)

derefinement = arguments["--derefinement"]

res_effective = int(N_gas ** (1.0 / 3.0) + 0.5)
phimode = float(arguments["--phimode"])

filename = (
    "M%3.2g_" % (M_gas)
    + ("Mstar%g_" % (M_star) if M_star > 0 else "")
    + ("rho_exp%g_" % (-density_exponent) if density_exponent < 0 else "")
    + "R%g_Z%g_S%g_A%g_B%g_I%g_Res%d_n%d_sol%g"
    % (
        R,
        metallicity,
        spin,
        2 * turbulence,
        magnetic_field,
        ISRF,
        res_effective,
        minmode,
#        turb_slope,
        turb_sol,
    )
    + ("_%d" % seed)
    + (
        "_collision_%g_%g_%g_%s"
        % (impact_dist, impact_param, v_impact, impact_axis)
        if impact_dist > 0
        else ""
    )
    + ".hdf5"
)
filename = filename.replace("+", "").replace("e0", "e")
filename = "".join(filename.split())

delta_m = M_gas / N_gas
delta_m_solar = delta_m / mass_unit
rho_avg = 3 * M_gas / R**3 / (4 * np.pi)
if delta_m_solar < 0.1:  # if we're doing something marginally IMF-resolving
    softening = (
        3.11e-5  # ~6.5 AU, minimum sink radius is 2.8 times that (~18 AU)
    )
    ncrit = 1e13  # ~100x the opacity limit
else:  # something more FIRE-like, where we rely on a sub-grid prescription turning gas into star particles
    softening = 0.1
    ncrit = 100

if fixed_ncrit:
    ncrit = fixed_ncrit

tff = (3 * np.pi / (32 * G * rho_avg)) ** 0.5
L = (4 * np.pi * R**3 / 3) ** (1.0 / 3)  # volume-equivalent box size
vrms = (6 / 5 * G * M_gas / R) ** 0.5 * turbulence**0.5

if turbulence:
    tcross = L / vrms
else:
    tcross = tff

turbenergy = (
    0.019111097819633344 * vrms**3 / L
)  # ST_Energy sets the dissipation rate of SPECIFIC energy ~ v^2 / (L/v) ~ v^3/L

paramsfile = str(
    open(
        os.path.realpath(__file__).replace("MakeCloud.py", "params.txt"), "r"
    ).read()
)

jet_particle_mass = min(delta_m, max(1e-4, delta_m / 10.0))
MS_wind_particle_mass = (
    jet_particle_mass / 10
)  # MS winds have lower mdot than jets, so we should be able to better resolve them this way

replacements = {
    "NAME": filename.replace(".hdf5", ""),
    "DTSNAP": tff / nsnap,
    "MAXTIMESTEP": tff / (nsnap),
    "SOFTENING": softening,
    "GASSOFT": 2.0e-8,
    "TMAX": tff * tmax,
    "RHOMAX": ncrit,
    "BOXSIZE": boxsize,
    "OUTFOLDER": "output",
    "JET_PART_MASS": jet_particle_mass,
    "MS_WIND_PART_MASS": MS_wind_particle_mass,
    "BH_SEED_MASS": delta_m / 2.0,
    "TURBDECAY": tcross / 2,
    "TURBENERGY": turbenergy,
    "TURBFREQ": tcross / 20,
    "TURB_KMIN": int(100 * 2 * np.pi / L) / 100.0,
    "TURB_KMAX": int(100 * 4 * np.pi / (L) + 1) / 100.0,
    "TURB_SIGMA": (M_gas/2e4)**0.5 * (R/10)**-0.5 * 600 * turbulence**0.5,
    "TURB_MINLAMBDA": int(100 * R / 2) / 100,
    "TURB_MAXLAMBDA": int(100 * R * 2) / 100,
    "TURB_COHERENCE_TIME": tcross / 2,
    "UNIT_L": 3.085678e18 * length_unit,
    "UNIT_M": 1.989e33 * mass_unit,
    "UNIT_V": v_unit * 1e2,
    "UNIT_B": B_unit,
    "ZINIT": metallicity,
    "ISRF": ISRF,
}

for k, r in replacements.items():
    paramsfile = paramsfile.replace(
        k, (r if isinstance(r, str) else "{:.2e}".format(r))
    )

open("params_" + filename.replace(".hdf5", "") + ".txt", "w").write(paramsfile)
if makebox:
    replacements_box = replacements.copy()
    replacements_box["NAME"] = filename.replace(".hdf5", "_BOX")
    replacements_box["BOXSIZE"] = L
    replacements_box["TURB_MINLAMBDA"] = int(100 * L / 2) / 100
    replacements_box["TURB_MAXLAMBDA"] = int(100 * L * 2) / 100
    paramsfile = str(
        open(
            os.path.realpath(__file__).replace("MakeCloud.py", "params.txt"),
            "r",
        ).read()
    )
    for k in replacements_box.keys():
        paramsfile = paramsfile.replace(k, str(replacements_box[k]))
    open("params_" + filename.replace(".hdf5", "") + "_BOX.txt", "w").write(
        paramsfile
    )
if makecylinder:
    # Get cylinder params
    R_cyl = R * np.sqrt(
        np.pi / (4 * cyl_aspect_ratio)
    )  # surface density equivalent cylinder
    L_cyl = R_cyl * 2 * cyl_aspect_ratio
    vrms_cyl = (
        2 * G * M_gas / L_cyl
    ) ** 0.5 * turbulence**0.5  # the potential is different for a cylinder than for a sphere, so we need to rescale vrms to get the right alpha, using E_grav_cyl = -GM**2/L
    vrms_cyl *= 0.71  # additional scaling found numerically to make the stirring run reproduce the right alpha and filament length (similarly determined numerical factor added to GIZMO)
    tcross_cyl = 2 * R_cyl / vrms_cyl
    boxsize_cyl = (
        L_cyl * 1.5 + R_cyl * 5
    )  # the box should fit the cylinder and be many times bigger than its width
    print(
        "Cylinder params: L=%g R=%g boxsize=%g vrms=%g"
        % (L_cyl, R_cyl, boxsize_cyl, vrms_cyl)
    )
    replacements_cyl = replacements.copy()
    replacements_cyl["NAME"] = filename.replace(".hdf5", "_CYL")
    replacements_cyl["BOXSIZE"] = boxsize_cyl
    # New driving params
    replacements_cyl["TURB_MINLAMBDA"] = int(100 * R_cyl) / 100
    replacements_cyl["TURB_MAXLAMBDA"] = int(100 * R_cyl * 4) / 100
    replacements_cyl["TURB_SIGMA"] = vrms_cyl
    replacements_cyl["TURB_COHERENCE_TIME"] = tcross_cyl / 2
    # Legacy driving params, probably needs tuning
    replacements_cyl["TURBDECAY"] = tcross_cyl / 2
    replacements_cyl["TURBENERGY"] = 0.019111097819633344 * vrms_cyl**3 / R_cyl
    replacements_cyl["TURBFREQ"] = tcross_cyl / 20
    replacements_cyl["TURB_KMIN"] = int(100 * 2 * np.pi / R_cyl) / 100.0
    replacements_cyl["TURB_KMAX"] = int(100 * 4 * np.pi / (R_cyl) + 1) / 100.0
    paramsfile = str(
        open(
            os.path.realpath(__file__).replace("MakeCloud.py", "params.txt"),
            "r",
        ).read()
    )
    for k in replacements_cyl.keys():
        paramsfile = paramsfile.replace(k, str(replacements_cyl[k]))
    open("params_" + filename.replace(".hdf5", "") + "_CYL.txt", "w").write(
        paramsfile
    )

if param_only:
    print("Parameters only run, exiting...")
    exit()

dm = M_gas / N_gas
mgas = np.repeat(dm, N_gas)

x = get_glass_coords(N_gas, glass_path)
Nx = len(x)
x = 2 * (x - 0.5)
print("Computing radii...")
r = cdist(x, [np.zeros(3)])[:, 0]
print("Done! Sorting coordinates...")
x = x[r.argsort()][:N_gas]
print("Done! Rescaling...")
x *= (float(Nx) / N_gas * 4 * np.pi / 3 / 8) ** (1.0 / 3) * R
print("Done! Recomupting radii...")
r = cdist(x, [np.zeros(3)])[:, 0]
x, r = x / r.max(), r / r.max()
print("Doing density profile...")
rnew = r ** (3.0 / (3 + density_exponent)) * R
x = x * (rnew / r)[:, None]
r = np.sum(x**2, axis=1) ** 0.5
r_order = r.argsort()
x, r = np.take(x, r_order, axis=0), r[r_order]

if not os.path.exists(turb_path):
    os.makedirs(turb_path)
fname = turb_path + "/vturb%d_beta%g_sol%g_seed%d.npy" % (minmode, turb_slope, turb_sol, seed)
if not os.path.isfile(fname):
    vt = TurbField(minmode=minmode, slope = turb_slope, sol_weight=turb_sol, seed=seed)
    nmin, nmax = vt.shape[-1] // 4, 3 * vt.shape[-1] // 4
    vt = vt[
        :, nmin:nmax, nmin:nmax, nmin:nmax
    ]  # we take the central cube of size L/2 so that opposide sides of the cloud are not correlated
    np.save(fname, vt)
else:
    vt = np.load(fname)

xgrid = np.linspace(-R, R, vt.shape[-1])
v = []
for i in range(3):
    v.append(interpolate.interpn((xgrid, xgrid, xgrid), vt[i, :, :, :], x))
v = np.array(v).T
print("Coordinates obtained!")

Mr = mgas.cumsum()
ugrav = G * np.sum(Mr / r * mgas)
v -= np.average(v, axis=0)
Eturb = 0.5 * M_gas / N_gas * np.sum(v**2)
v *= np.sqrt(turbulence * ugrav / Eturb)
E_rot_target = spin * ugrav
Rcyl = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
omega = Rcyl**omega_exponent
vrot = np.cross(np.c_[np.zeros_like(omega), np.zeros_like(omega), omega], x)
Erot_actual = np.sum(0.5 * mgas[:, None] * vrot**2)
vrot *= np.sqrt(E_rot_target / Erot_actual)
v += vrot

B = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.ones(N_gas)]
vA_unit = (
    3.429e8
    * B_unit
    * (M_gas) ** -0.5
    * R**1.5
    * np.sqrt(4 * np.pi / 3)
    / v_unit
)  # alfven speed for unit magnetic field
uB = (
    0.5 * M_gas * vA_unit**2
)  # magnetic energy we would have for unit magnetic field
if bfixed > 0:
    B = B * bfixed
else:
    B = B * np.sqrt(
        magnetic_field * ugrav / uB
    )  # renormalize to desired magnetic energy

v = v - np.average(v, axis=0)
x = x - np.average(x, axis=0)

r, phi = np.sum(x**2, axis=1) ** 0.5, np.arctan2(x[:, 1], x[:, 0])
theta = np.arccos(x[:, 2] / r)
phi += phimode * np.sin(2 * phi) / 2
x = (
    r[:, np.newaxis]
    * np.c_[
        np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    ]
)

if makecylinder:

    def ind_in_cylinder(x, L_cyl, R_cyl):
        return (np.abs(x[:, 0]) < L_cyl / 2) & (
            np.sum(x[:, 1:] ** 2, axis=1) < R_cyl**2
        )

    # Just get a roughly homogeneous cylinder along the x axis, we will stir it anyway
    N_cyl = 0
    while (
        N_cyl <= N_gas
    ):  # should be very unlikely that we need to repeat, but let's check to be sure
        x_cyl = np.random.rand(2 * N_gas, 3) * 2 - 1
        x_cyl[:, 0] *= L_cyl / 2
        x_cyl[:, 1] *= R_cyl
        x_cyl[:, 2] *= R_cyl
        x_cyl = x_cyl[ind_in_cylinder(x_cyl, L_cyl, R_cyl)]
        N_cyl = len(x_cyl)
        # print("N_cyl: %g N_gas: %g"%(N_cyl,N_gas))
    x_cyl = x_cyl[:N_gas]  # keep only the right amount of gas
    # Let's add some initial velocity to make the driving phase shorter, let's start with a rotational component
    v_cyl = np.cross([1, 0, 0], x_cyl, axis=-1) / R_cyl
    # tangential with magnitude increasing linearly
    v_cyl *= vrms_cyl

u = np.ones_like(mgas) * 0.101 / 2.0  # /2 needed because it is molecular

if impact_dist > 0:
    x = np.concatenate([x, x])
    impact_dir = {
        "x": np.array([1.0, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }[impact_axis]
    impact_param_dir = {
        "x": np.array([0, 1, 0]),
        "y": np.array([0, 0, 1]),
        "z": np.array([1, 0, 0]),
    }[impact_axis]
    x[:N_gas] += impact_dist * R * impact_dir
    x[N_gas:] -= impact_dist * R * impact_dir
    x[:N_gas] += 0.5 * impact_param * R * impact_param_dir
    x[N_gas:] -= 0.5 * impact_param * R * impact_param_dir
    v = np.concatenate([v, v])
    vrms = np.sum(v**2, axis=1).mean() ** 0.5
    v[:N_gas] -= v_impact * vrms * impact_dir
    v[N_gas:] += v_impact * vrms * impact_dir
    B = np.concatenate([B, B])
    u = np.concatenate([u, u])
    mgas = np.concatenate([mgas, mgas])

u = (
    np.ones_like(mgas) * (200 / v_unit) ** 2
)  # start with specific internal energy of (200m/s)^2, this is overwritten unless starting with restart flag 2###### #0.101/2.0 #/2 needed because it is molecular

if diffuse_gas:
    # assuming 10K vs 10^4K gas: factor of ~10^3 density contrast
    rho_warm = M_gas * 3 / (4 * np.pi * R**3) / 1000
    M_warm = (
        boxsize**3 - (4 * np.pi * R**3 / 3)
    ) * rho_warm  # mass of diffuse box-filling medium
    N_warm = int(M_warm / (M_gas / N_gas))
    if derefinement:
        x0 = get_glass_coords(N_gas, glass_path)
        Nx = len(x0)
        x0 = 2 * (x0 - 0.5)
        r0 = (x0 * x0).sum(1) ** 0.5
        x0, r0 = x0[r0.argsort()], r0[r0.argsort()]
        # first lay down the stuff within 3*R
        N_warm = int(
            4 * np.pi * rho_warm * (3 * R) ** 3 / 3 / dm
        )  # number of cells within 3R
        x_warm = (
            x0[:N_warm] * 3 * R / r0[N_warm - 1]
        )  # uniform density of cells within 3R
        x0 = x0[
            N_warm:
        ]  # now we take the ones outside the initial sphere and map them to a n(R) ~ R^-3 profile so that we get constant number of cells per log radius interval
        r0 = r0[N_warm:]
        rnew = 3 * R * np.exp(np.arange(len(x0)) / N_warm / 3)
        x_warm = np.concatenate([x_warm, (rnew / r0)[:, None] * x0], axis=0)
        x_warm = x_warm[np.max(np.abs(x_warm), axis=1) < boxsize / 2]
        N_warm = len(x_warm)
        R_warm = (x_warm * x_warm).sum(1) ** 0.5
        mgas = np.concatenate(
            [mgas, np.clip(dm * (R_warm / (3 * R)) ** 3, dm, np.inf)]
        )
    else:
        x_warm = boxsize * np.random.rand(N_warm, 3) - boxsize / 2
        if impact_dist == 0:
            x_warm = x_warm[np.sum(x_warm**2, axis=1) > R**2]
        N_warm = len(x_warm)
        mgas = np.concatenate(
            [mgas, np.repeat(mgas.sum() / len(mgas), N_warm)]
        )
    x = np.concatenate([x, x_warm])
    v = np.concatenate([v, np.zeros((N_warm, 3))])
    Bmag = np.average(np.sum(B**2, axis=1)) ** 0.5
    B = np.concatenate(
        [B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([0, 0, 1])]
    )
    u = np.concatenate([u, np.repeat(101.0, N_warm)])

    if makecylinder:
        # The magnetic field is paralell to the cylinder (true at low densities, so probably fine for IC)
        B_cyl = np.concatenate(
            [B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([1, 0, 0])]
        )
        # Add diffuse medium
        M_warm_cyl = (boxsize_cyl**3 - (4 * np.pi * R**3 / 3)) * rho_warm
        N_warm_cyl = int(M_warm_cyl / (M_gas / N_gas))
        x_warm = (
            boxsize_cyl * np.random.rand(N_warm_cyl, 3) - boxsize_cyl / 2
        )  # will be recentered later
        x_warm = x_warm[
            ~ind_in_cylinder(x_warm, L_cyl, R_cyl)
        ]  # keep only warm gas outside the cylinder
        # print("N_warm_cyl: %g N_warm_cyl_kept %g "%(N_warm_cyl,len(x_warm)))
        N_warm_cyl = len(x_warm)
        x_cyl = np.concatenate([x_cyl, x_warm])
        v_cyl = np.concatenate([v_cyl, np.zeros((N_warm, 3))])

else:
    N_warm = 0

rho = np.repeat(3 * M_gas / (4 * np.pi * R**3), len(mgas))
if diffuse_gas:
    rho[-N_warm:] /= 1000
h = (32 * mgas / rho) ** (1.0 / 3)

x += boxsize / 2  # cloud is always centered at (boxsize/2,boxsize/2,boxsize/2)
if makecylinder:
    x_cyl += boxsize_cyl / 2

print("Writing snapshot...")

F = h5py.File(filename, "w")
F.create_group("PartType0")
F.create_group("Header")
F["Header"].attrs["NumPart_ThisFile"] = [
    len(mgas),
    0,
    0,
    0,
    0,
    (1 if M_star > 0 else 0),
]
F["Header"].attrs["NumPart_Total"] = [
    len(mgas),
    0,
    0,
    0,
    0,
    (1 if M_star > 0 else 0),
]
F["Header"].attrs["BoxSize"] = boxsize
F["Header"].attrs["Time"] = 0.0
F["PartType0"].create_dataset("Masses", data=mgas)
F["PartType0"].create_dataset("Coordinates", data=x)
F["PartType0"].create_dataset("Velocities", data=v)
F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(len(mgas)))
F["PartType0"].create_dataset("InternalEnergy", data=u)

if M_star > 0:
    F.create_group("PartType5")
    # Let's add the sink at the center
    F["PartType5"].create_dataset("Masses", data=np.array([M_star]))
    F["PartType5"].create_dataset(
        "Coordinates", data=[x_star]
    )  # at the center
    F["PartType5"].create_dataset("Velocities", data=[v_star])  # at rest
    F["PartType5"].create_dataset(
        "ParticleIDs", data=np.array([F["PartType0/ParticleIDs"][:].max() + 1])
    )
    # Advanced properties for sinks
    F["PartType5"].create_dataset(
        "BH_Mass", data=M_star
    )  # all the mass in the sink/protostar/star
    F["PartType5"].create_dataset(
        "BH_Mass_AlphaDisk", data=np.array([0.0])
    )  # starts with no disk
    F["PartType5"].create_dataset(
        "BH_Mdot", data=np.array([0.0])
    )  # starts with no mdot
    F["PartType5"].create_dataset(
        "BH_Specific_AngMom", data=np.array([0.0])
    )  # starts with no angular momentum
    F["PartType5"].create_dataset(
        "SinkRadius", data=np.array([softening])
    )  # Sinkradius set to softening
    F["PartType5"].create_dataset("StellarFormationTime", data=np.array([0.0]))
    F["PartType5"].create_dataset("ProtoStellarAge", data=np.array([0.0]))
    F["PartType5"].create_dataset(
        "ProtoStellarStage", data=np.array([5], dtype=np.int32), dtype=np.int32
    )
    # Stellar properties
    # if (central_star or central_SN):
    # if central_star:
    #       print("Assuming central sink is a ZAMS star")
    # starts as ZAMS star
    # else:
    # print("Assuming central sink is a ZAMS star about to go supernova")
    # F["PartType5"].create_dataset("ProtoStellarStage", data=np.array([6],dtype=np.int32), dtype=np.int32) #starts as ZAMS star going SN
    # Set guess for ZAMS stellar radius, will be overwritten
    if (M_star) > 1.0:
        R_ZAMS = (M_star) ** 0.57
    else:
        R_ZAMS = (M_star) ** 0.8
    F["PartType5"].create_dataset(
        "ProtoStellarRadius_inSolar", data=np.array([R_ZAMS])
    )  # Sinkradius set to softening
    F["PartType5"].create_dataset(
        "StarLuminosity_Solar", data=np.array([0.0])
    )  # dummy
    F["PartType5"].create_dataset("Mass_D", data=np.array([0.0]))  # No D left

if magnetic_field > 0.0:
    F["PartType0"].create_dataset("MagneticField", data=B)
F.close()

if makebox:
    F = h5py.File(filename.replace(".hdf5", "_BOX.hdf5"), "w")
    F.create_group("PartType0")
    F.create_group("Header")
    F["Header"].attrs["NumPart_ThisFile"] = [len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["NumPart_Total"] = [len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["MassTable"] = [M_gas / len(mgas), 0, 0, 0, 0, 0]
    F["Header"].attrs["BoxSize"] = (4 * np.pi * R**3 / 3) ** (1.0 / 3)
    F["Header"].attrs["Time"] = 0.0
    F["PartType0"].create_dataset("Masses", data=mgas[: len(mgas)])
    F["PartType0"].create_dataset(
        "Coordinates",
        data=np.random.rand(len(mgas), 3) * F["Header"].attrs["BoxSize"],
    )
    F["PartType0"].create_dataset("Velocities", data=np.zeros((len(mgas), 3)))
    F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(len(mgas)))
    F["PartType0"].create_dataset("InternalEnergy", data=u)
    if magnetic_field > 0.0:
        F["PartType0"].create_dataset("MagneticField", data=B[: len(mgas)])
    F.close()

if makecylinder:
    F = h5py.File(filename.replace(".hdf5", "_CYL.hdf5"), "w")
    F.create_group("PartType0")
    F.create_group("Header")
    F["Header"].attrs["NumPart_ThisFile"] = [N_gas + N_warm_cyl, 0, 0, 0, 0, 0]
    F["Header"].attrs["NumPart_Total"] = [N_gas + N_warm_cyl, 0, 0, 0, 0, 0]
    F["Header"].attrs["MassTable"] = [M_gas / N_gas, 0, 0, 0, 0, 0]
    F["Header"].attrs["BoxSize"] = boxsize_cyl
    F["Header"].attrs["Time"] = 0.0
    F["PartType0"].create_dataset("Masses", data=mgas)
    F["PartType0"].create_dataset("Coordinates", data=x_cyl)
    F["PartType0"].create_dataset("Velocities", data=v_cyl)
    F["PartType0"].create_dataset(
        "ParticleIDs", data=1 + np.arange(N_gas + N_warm_cyl)
    )
    F["PartType0"].create_dataset("InternalEnergy", data=u)
    if magnetic_field > 0.0:
        F["PartType0"].create_dataset("MagneticField", data=B_cyl)
    F.close()
