# MakeCloud

Requires the following packages:
numpy
scipy
h5py
docopt

If you want to use glassy initial conditions, you will need <a href=http://www.tapir.caltech.edu/~mgrudich/files/glass_orig.npy>this file</a> in the same path as MakeCloud.py.

If you want the random Gaussian velocity/magnetic fields, you need to download and extract <a href=http://www.tapir.caltech.edu/~mgrudich/files/turb.tar.gz> this.</a> You should have a directory called turb in the same path as MakeCloud.py.

#Usage 

python MakeCloud.py [options]
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
