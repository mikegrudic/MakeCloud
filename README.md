# MakeCloud

Requires the following packages:
numpy
scipy
h5py
docopt

If you want to use glassy initial conditions, you will need <a href=http://www.tapir.caltech.edu/~mgrudich/glass_orig.npy>this file</a>, and you should point the script to it via the --glass_path option.

By default, we generate turbulent velocity fields on the fly but then store that data somewhere so we don't have to do all those FFTs again next. You'll have to specify the path where the files get stored via the --turb_path option.

#Usage

Run python MakeCloud.py -h for instructions

e.g. if I wanted a 10^6 solar mass cloud of radius 100pc, resolved in 10^6 gas cells, I would do

python MakeCloud.py --M=1e6 --R=100 --N=1000000
