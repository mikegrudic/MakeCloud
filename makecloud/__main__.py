"""
Usage: MakeCloud [options]

Options:
   -h --help            Show this screen.
   --R=<pc>             Outer radius of the cloud in pc [default: 10.0]
   --M=<msun>           Mass of the cloud in msun [default: None]
   --N=<N>              Number of gas particles [default: 2000000]
   --nH=<cm^-3>         Mean cloud hydrogen number density; overrides M if set
   --dm=<msun>          Target particle mass in msun; overrides N if set
   --density_exponent=<f>   Power law exponent of the density profile [default: 0.0]
   --spin=<f>           Spin parameter: fraction of binding energy in solid-body rotation [default: 0.0]
   --omega_exponent=<f>  Powerlaw exponent of rotational frequency as a function of cylindrical radius [default: 0.0]
   --turb_slope=<f>      Slope of the turbulent power spectra [default: 2.0]
   --turb_sol=<f>       Fraction of turbulence in solenoidal modes [default: 0.5]
   --alpha_turb=<f>     Turbulent virial parameter (BM92 convention: 2Eturb/|Egrav|) [default: 2.]
   --bturb=<f>          Magnetic energy as a fraction of the binding energy [default: 0.1]
   --bfixed=<f>         Deprecated: use --B instead [default: 0]
   --B=<Bx,By,Bz>      Magnetic field in code units: 1 value sets magnitude along z, 3 comma-separated values set the full vector. Overrides bturb and bfixed.
   --minmode=<N>        Minimum populated turbulent wavenumber for Gaussian initial velocity field, in units of pi/R [default: 2]
   --turb_path=<name>   Path to store turbulent velocity fields so that we only need to generate them once (defaults to ~/turb)
   --glass_path=<name>  Contains the the path of the glass file (defaults to your home directory)
   --boxsize=<f>        Simulation box size
   --Mstar=<msun>       Mass of the star/black hole, if any [default: 0.0]
   --star_list=<file>   ASCII file with one star per row: mass(Msun) age(code) x y z (code). Overrides --Mstar/--star_age/--x_star. [default: None]
   --v_star=<vx,vy,vz>  Velocity of the star [default: 0.0,0.0,0.0]
   --x_star=<x,y,z>     Position of the star, defaults to center of the box
   --star_stage=<N>     Evolutionary stage of the star/black hole [default: 5]
   --star_age=<t>       Age of the star in code units [default: 0.0]
   --derefinement       Apply radial derefinement to ambient cells outside of 3* cloud radius
   --no_diffuse_gas     Remove diffuse ISM envelope fills the rest of the box with uniform density.
   --density_contrast=<f>  Density contrast between cloud and diffuse ISM [default: 1000]
   --phimode=<f>        Relative amplitude of m=2 density perturbation (e.g. for Boss-Bodenheimer test) [default: 0.0]
   --B_unit=<gauss>     Unit of magnetic field in gauss [default: 1.0]
   --length_unit=<pc>   Unit of length in pc [default: 1]
   --mass_unit=<msun>   Unit of mass in M_sun [default: 1]
   --v_unit=<m/s>       Unit of velocity in m/s [default: 1e3]
   --unit_system=<name> Units system to adopt (options: starforge_classic (m/s - pc - Msun - T), FIRE (km/s - kpc - 1e10Msun - uG)) [default: None]
   --turb_seed=<N>      Random seed for turbulence initialization [default: 42]
   --tmax=<N>           Maximum time to run the simulation to, in units of the freefall time [default: 5]
   --nsnap=<N>          Number of snapshots per freefall time [default: 150]
   --param_only         Just makes the parameters file, not the IC
   --fixed_ncrit=<f>    Fixes ncrit to a specific value [default: 0.0]
   --makebox            Creates a second box IC of equivalent volume and mass to the cloud
   --L=<pc>             Box side length for --makebox, overrides --R (half-length = L/2) [default: None]
   --impact_dist=<b>    Initial separation between cloud centers of mass in units of the cloud radius  (0 is no cloud-cloud collision) [default: 0.0]
   --impact_param=<b>   Impact parameter of cloud-cloud collision in units of the cloud radius [default: 0.0]
   --v_impact=<v>       Impact velocity, in units of the cloud's RMS turbulent velocity [default: 1.0]
   --impact_axis=<x>    Axis along which collision occurs (z is along magnetic field lines) [default: x]
   --makecylinder       Creates a third, cylindrical IC of equivalent volume and mass to the cloud
   --cyl_aspect_ratio=<f>   Sets the aspect ratio of the cylinder, i.e. Length/Diameter [default: 10]
   --Z=<solar>          Metallicity of the cloud in Solar units (just for params file) [default: 1.0]
   --ISRF=<solar>       Interstellar radiation background of the cloud in Solar neighborhood units (just for params file) [default: 1.0]
"""

from docopt import docopt
from . import MakeCloud


def main():
    kwargs = {k[2:]: v for k, v in docopt(__doc__).items() if k != "--help"}
    if kwargs["unit_system"] == "None":
        kwargs["unit_system"] = None
    if kwargs["star_list"] == "None":
        kwargs["star_list"] = None
    if kwargs["L"] == "None":
        kwargs["L"] = None
    if kwargs["M"] == "None":
        kwargs["M"] = None
    kwargs["Lbox"] = kwargs.pop("L")
    if float(kwargs["bfixed"]) != 0.0:
        import warnings
        warnings.warn("--bfixed is deprecated; use --B instead", DeprecationWarning, stacklevel=2)
    if kwargs["B"] is not None:
        vals = [float(v) for v in kwargs["B"].split(",")]
        if len(vals) == 1:
            kwargs["B_fixed"] = vals[0]
        elif len(vals) == 3:
            kwargs["B_fixed"] = vals
        else:
            raise ValueError("--B must be 1 or 3 comma-separated values")
    del kwargs["B"]
    MakeCloud(**kwargs)


if __name__ == "__main__":
    main()
