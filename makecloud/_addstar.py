"""
Usage: AddStarToSnapshot <snapshot> [options]

Options:
   -h --help            Show this screen.
   --Mstar=<msun>       Mass of the star/black hole [default: 1.0]
   --v_star=<vx,vy,vz>  Velocity of the star [default: 0.0,0.0,0.0]
   --x_star=<x,y,z>     Position of the star (defaults to center of the box)
   --star_stage=<N>     Evolutionary stage of the star/black hole [default: 5]
   --star_age=<t>       Age of the star in code units [default: 0.0]
   --star_list=<file>   ASCII file with columns: mass age x y z [vx vy vz]
"""

import numpy as np
import h5py
from docopt import docopt


def AddStarToSnapshot(
    snapshot,
    Mstar=1.0,
    v_star=None,
    x_star=None,
    star_stage=5,
    star_age=0.0,
    star_list=None,
):
    Mstar = float(Mstar)
    star_stage = int(float(star_stage))
    star_age = float(star_age)

    if star_list is not None:
        sl = np.atleast_2d(np.loadtxt(star_list))
        pt5_masses = sl[:, 0]
        pt5_ages = sl[:, 1]
        pt5_coords = sl[:, 2:5]
        pt5_vels = sl[:, 5:8] if sl.shape[1] >= 8 else np.zeros((len(pt5_masses), 3))
    else:
        if v_star is None:
            v_star = np.zeros(3)
        elif isinstance(v_star, str):
            v_star = np.array([float(v) for v in v_star.split(",")])
        else:
            v_star = np.asarray(v_star, dtype=float)

        with h5py.File(snapshot, "r") as F:
            boxsize = F["Header"].attrs["BoxSize"]

        if x_star is None:
            x_star = np.full(3, 0.5 * boxsize)
        elif isinstance(x_star, str):
            x_star = np.array([float(v) for v in x_star.split(",")])
        else:
            x_star = np.asarray(x_star, dtype=float)

        pt5_masses = np.array([Mstar])
        pt5_ages = np.array([star_age])
        pt5_coords = np.array([x_star])
        pt5_vels = np.array([v_star])

    N_new = len(pt5_masses)
    R_ZAMS = np.where(pt5_masses > 1.0, pt5_masses**0.57, pt5_masses**0.8)

    with h5py.File(snapshot, "a") as F:
        # determine softening from gas particle mass
        if "PartType0" in F and len(F["PartType0/Masses"]) > 0:
            dm_solar = float(np.mean(F["PartType0/Masses"][:]))
            softening = 3.11e-5 if dm_solar < 0.1 else 0.1
        else:
            softening = 0.1

        # find max existing particle ID for offsetting new IDs
        max_id = 0
        for ptype in F:
            if ptype.startswith("PartType") and "ParticleIDs" in F[ptype]:
                max_id = max(max_id, int(F[ptype]["ParticleIDs"][:].max()))

        if "PartType5" in F:
            # append to existing stars
            pt5 = F["PartType5"]
            for key, new_data in [
                ("Masses", pt5_masses),
                ("Coordinates", pt5_coords),
                ("Velocities", pt5_vels),
                ("BH_Mass", pt5_masses),
                ("BH_Mass_AlphaDisk", np.zeros(N_new)),
                ("BH_Mdot", np.zeros(N_new)),
                ("BH_Specific_AngMom", np.zeros(N_new)),
                ("SinkRadius", np.full(N_new, softening)),
                ("StellarFormationTime", -pt5_ages),
                ("ProtoStellarAge", pt5_ages),
                ("ProtoStellarRadius_inSolar", R_ZAMS),
                ("StarLuminosity_Solar", np.zeros(N_new)),
                ("Mass_D", np.zeros(N_new)),
                ("ParticleIDs", max_id + 1 + np.arange(N_new)),
            ]:
                old = pt5[key][:]
                del pt5[key]
                pt5.create_dataset(key, data=np.concatenate([old, new_data]))

            new_ids_arr = np.full(N_new, star_stage, dtype=np.int32)
            old_stage = pt5["ProtoStellarStage"][:]
            del pt5["ProtoStellarStage"]
            pt5.create_dataset(
                "ProtoStellarStage",
                data=np.concatenate([old_stage, new_ids_arr]),
                dtype=np.int32,
            )
            N_total = len(pt5["Masses"])
        else:
            pt5 = F.create_group("PartType5")
            pt5.create_dataset("Masses", data=pt5_masses)
            pt5.create_dataset("Coordinates", data=pt5_coords)
            pt5.create_dataset("Velocities", data=pt5_vels)
            pt5.create_dataset("ParticleIDs", data=max_id + 1 + np.arange(N_new))
            pt5.create_dataset("BH_Mass", data=pt5_masses)
            pt5.create_dataset("BH_Mass_AlphaDisk", data=np.zeros(N_new))
            pt5.create_dataset("BH_Mdot", data=np.zeros(N_new))
            pt5.create_dataset("BH_Specific_AngMom", data=np.zeros(N_new))
            pt5.create_dataset("SinkRadius", data=np.full(N_new, softening))
            pt5.create_dataset("StellarFormationTime", data=-pt5_ages)
            pt5.create_dataset("ProtoStellarAge", data=pt5_ages)
            pt5.create_dataset(
                "ProtoStellarStage",
                data=np.full(N_new, star_stage, dtype=np.int32),
                dtype=np.int32,
            )
            pt5.create_dataset("ProtoStellarRadius_inSolar", data=R_ZAMS)
            pt5.create_dataset("StarLuminosity_Solar", data=np.zeros(N_new))
            pt5.create_dataset("Mass_D", data=np.zeros(N_new))
            N_total = N_new

        header = F["Header"]
        counts = header.attrs["NumPart_ThisFile"].copy()
        counts[5] = N_total
        header.attrs["NumPart_ThisFile"] = counts
        counts_total = header.attrs["NumPart_Total"].copy()
        counts_total[5] = N_total
        header.attrs["NumPart_Total"] = counts_total


def main():
    args = docopt(__doc__)
    snapshot = args.pop("<snapshot>")
    kwargs = {k[2:]: v for k, v in args.items() if k != "--help"}
    if kwargs["x_star"] == "None" or kwargs["x_star"] is None:
        kwargs["x_star"] = None
    AddStarToSnapshot(snapshot, **kwargs)


if __name__ == "__main__":
    main()
