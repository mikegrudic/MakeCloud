__all__ = ["MakeCloud", "TurbField", "get_glass_coords"]

import os
import numpy as np
from scipy import fftpack, interpolate
from scipy.spatial.distance import cdist
import h5py


def _read_params_template():
    from importlib.resources import files

    return files("makecloud").joinpath("params.txt").read_text()


def _get_box_glass(N, cache_dir):
    import urllib.request

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    cbrt = round(N ** (1 / 3))
    path = os.path.join(cache_dir, "glass_%d.hdf5" % cbrt)
    if not os.path.exists(path):
        print("Downloading glass_%d.hdf5..." % cbrt)
        urllib.request.urlretrieve(
            "https://users.flatironinstitute.org/~mgrudic/glass/glass_%d.hdf5" % cbrt,
            path,
        )
    x = h5py.File(path)["Coordinates"][:]
    while len(x) < N:
        x = np.concatenate([
            x / 2 + np.array([i * 0.5, j * 0.5, k * 0.5])
            for i in range(2) for j in range(2) for k in range(2)
        ])
    return x[:N]


def get_glass_coords(N_gas, glass_path, center_on_cell=False):
    x = h5py.File(glass_path)["Coordinates"][:]
    if not center_on_cell:
        np.random.seed(42)
        x = (x + np.random.rand(x.shape[1])) % 1.0
    while len(x) * np.pi * 4 / 3 / 8 < N_gas:
        print(
            "Need %d particles, have %d. Tessellating 8 copies of the glass file to get required particle number"
            % (N_gas * 8 / (4 * np.pi / 3), len(x))
        )
        x = np.concatenate(
            [
                x / 2 + i * np.array([0.5, 0, 0]) + j * np.array([0, 0.5, 0]) + k * np.array([0, 0, 0.5])
                for i in range(2)
                for j in range(2)
                for k in range(2)
            ]
        )
    print("Glass loaded!")
    order = x.max(axis=1).argsort()
    return x[order]


def TurbField(res=256, minmode=2, maxmode=64, slope=2.0, sol_weight=1.0, seed=42):
    freqs = fftpack.fftfreq(res)
    freq3d = np.array(np.meshgrid(freqs, freqs, freqs, indexing="ij"))
    intfreq = np.around(freq3d * res)
    kSqr = np.sum(np.abs(freq3d) ** 2, axis=0)
    intkSqr = np.sum(np.abs(intfreq) ** 2, axis=0)
    VK = []

    for i in range(3):
        np.random.seed(seed + i)
        rand_phase = fftpack.fftn(np.random.normal(size=kSqr.shape))
        vk = rand_phase * (float(minmode) / res) ** 2 / (np.power(kSqr, slope / 2.0) + 1e-300)
        vk[intkSqr == 0] = 0.0
        vk[intkSqr < minmode**2] *= intkSqr[intkSqr < minmode**2] ** 2 / minmode**4
        vk *= np.exp(-intkSqr / maxmode**2)
        VK.append(vk)
    VK = np.array(VK)

    vk_new = np.zeros_like(VK)
    for i in range(3):
        for j in range(3):
            if i == j:
                vk_new[i] += sol_weight * VK[j]
            vk_new[i] += (1 - 2 * sol_weight) * freq3d[i] * freq3d[j] / (kSqr + 1e-300) * VK[j]
    vk_new[:, kSqr == 0] = 0.0
    VK = vk_new

    vel = np.array([fftpack.ifftn(vk).real for vk in VK])
    vel -= np.average(vel, axis=(1, 2, 3))[:, np.newaxis, np.newaxis, np.newaxis]
    vel = vel / np.sqrt(np.sum(vel**2, axis=0).mean())
    return np.array(vel)


def MakeCloud(
    R=10.0,
    M=None,
    N=2000000,
    nH=None,
    dm=None,
    density_exponent=0.0,
    spin=0.0,
    omega_exponent=0.0,
    turb_slope=2.0,
    turb_sol=0.5,
    alpha_turb=2.0,
    bturb=0.1,
    bfixed=0.0,
    B_fixed=None,
    minmode=2,
    turb_path=None,
    glass_path=None,
    boxsize=None,
    Mstar=0.0,
    v_star=None,
    x_star=None,
    star_stage=5,
    star_age=0.0,
    star_list=None,
    derefinement=False,
    no_diffuse_gas=False,
    density_contrast=1000.0,
    phimode=0.0,
    B_unit=1.0,
    length_unit=1.0,
    mass_unit=1.0,
    v_unit=1e3,
    unit_system=None,
    turb_seed=42,
    tmax=5,
    nsnap=150,
    param_only=False,
    fixed_ncrit=0.0,
    makebox=False,
    Lbox=None,
    impact_dist=0.0,
    impact_param=0.0,
    v_impact=1.0,
    impact_axis="x",
    makecylinder=False,
    cyl_aspect_ratio=10.0,
    Z=1.0,
    ISRF=1.0,
):
    """Generate a molecular cloud initial conditions file for GIZMO/GADGET.

    Parameters correspond directly to the CLI options of the MakeCloud command.
    Returns the filename of the generated IC file.
    """
    R = float(R)
    if nH is not None and M is not None:
        # Both nH and M given: infer R from them
        # nH = 3M/(4piR^3) * 0.71/m_H  =>  R [pc] = (3M/(4pi*nH*m_H/0.71) * Msun_g / (pc_cm)^3)^(1/3)
        R = (float(M) * 1.989e33 / (float(nH) * (1.673e-24 / 0.71)) * 3 / (4 * np.pi)) ** (1.0 / 3) / 3.086e18
        print("nH and M both specified: inferred R = %.3g pc" % R)
        M_gas = float(M)
    elif nH is not None and makebox and Lbox is not None:
        # makebox mode with nH: set R=L/2 and compute mass from box volume so the box IC has the correct mean density
        R = float(Lbox) / 2
        M_gas = float(nH) * (1.673e-24 / 0.71) * float(Lbox) ** 3 * (3.086e18) ** 3 / 1.989e33
        print("makebox nH mode: R=%.4g pc, box M = %.3g Msun (nH=%.3g in L=%.4g pc box)" % (R, M_gas, float(nH), float(Lbox)))
    elif nH is not None:
        # Only nH given: infer M from R and nH
        # nH = 3M/(4piR^3) * 0.71/m_H  =>  M [Msun] = nH * m_H/0.71 * (4pi/3) * R_cm^3 / Msun_g
        M_gas = float(nH) * (1.673e-24 / 0.71) * (4 * np.pi / 3) * (R * 3.086e18) ** 3 / 1.989e33
    elif M is not None:
        M_gas = float(M)
    else:
        M_gas = 2e4
    if dm is not None:
        N = M_gas / float(dm)
    N_gas = int(float(N) + 0.5)
    M_star = float(Mstar)

    if v_star is None:
        v_star = np.zeros(3)
    elif isinstance(v_star, str):
        v_star = np.array([float(v) for v in v_star.split(",")])
    else:
        v_star = np.asarray(v_star, dtype=float)

    spin = float(spin)
    omega_exponent = float(omega_exponent)
    turbulence = float(alpha_turb) / 2
    seed = int(float(turb_seed) + 0.5)
    tmax = int(float(tmax))
    nsnap = int(float(nsnap))
    turb_slope = float(turb_slope)
    turb_sol = float(turb_sol)
    magnetic_field = float(bturb)
    bfixed = float(bfixed)
    minmode = int(minmode)
    star_stage = int(float(star_stage))
    star_age = float(star_age)
    diffuse_gas = not no_diffuse_gas
    density_contrast = float(density_contrast)
    impact_param = float(impact_param)
    impact_dist = float(impact_dist)
    v_impact = float(v_impact)
    impact_axis = str(impact_axis)
    cyl_aspect_ratio = float(cyl_aspect_ratio)
    fixed_ncrit = float(fixed_ncrit)
    density_exponent = float(density_exponent)
    metallicity = float(Z)
    ISRF = float(ISRF)

    if unit_system is not None:
        match unit_system:
            case "starforge_classic":
                length_unit_pc = 1
                mass_unit_Msun = 1
                v_unit_SI = 1
                B_unit_gauss = 1e4
            case "FIRE":
                length_unit_pc = 1e3
                mass_unit_Msun = 1e10
                v_unit_SI = 1e3
                B_unit_gauss = 1
    else:
        length_unit_pc = float(length_unit)
        mass_unit_Msun = float(mass_unit)
        v_unit_SI = float(v_unit)
        B_unit_gauss = float(B_unit)

    G = 4300.71 * v_unit_SI**-2 * mass_unit_Msun / length_unit_pc

    if turb_path is None:
        turb_path = os.path.expanduser("~") + "/turb"
    if glass_path is None:
        prefix = os.path.expanduser("~") + "/.makecloud_glass"
        glass_path = prefix + "/glass_256.hdf5"
        if not os.path.exists(glass_path):
            if not os.path.isdir(prefix):
                os.mkdir(prefix)
            import urllib.request

            print("Downloading glass file...")
            urllib.request.urlretrieve(
                "https://users.flatironinstitute.org/~mgrudic/glass/glass_256.hdf5",
                glass_path,
            )

    if boxsize is None:
        boxsize = 10 * R
    else:
        boxsize = float(boxsize)

    if x_star is None:
        x_star = np.repeat(0.5 * boxsize, 3)
    elif isinstance(x_star, str):
        x_star = np.array([float(v) for v in x_star.split(",")])
    else:
        x_star = np.asarray(x_star, dtype=float)

    res_effective = int(N_gas ** (1.0 / 3.0) + 0.5)
    phimode = float(phimode)

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
            turb_sol,
        )
        + ("_%d" % seed)
        + (
            "_collision_%g_%g_%g_%s" % (impact_dist, impact_param, v_impact, impact_axis)
            if impact_dist > 0
            else ""
        )
        + ".hdf5"
    )
    filename = filename.replace("+", "").replace("e0", "e")
    filename = "".join(filename.split())

    dm = M_gas / N_gas
    dm_solar = dm / mass_unit_Msun
    rho_avg = 3 * M_gas / R**3 / (4 * np.pi)
    if dm_solar < 0.1:
        softening = 3.11e-5
        ncrit = 1e13
    else:
        softening = 0.1
        ncrit = 100

    if fixed_ncrit:
        ncrit = fixed_ncrit

    tff = (3 * np.pi / (32 * G * rho_avg)) ** 0.5
    L = (4 * np.pi * R**3 / 3) ** (1.0 / 3)
    vrms = (6 / 5 * G * M_gas / R) ** 0.5 * turbulence**0.5

    if turbulence:
        tcross = L / vrms
    else:
        tcross = tff

    turbenergy = 0.019111097819633344 * vrms**3 / L

    jet_particle_mass = min(dm, max(1e-4, dm / 10.0))
    MS_wind_particle_mass = jet_particle_mass / 10

    replacements = {
        "NAME": filename.replace(".hdf5", ""),
        "DTSNAP": tff / nsnap,
        "MAXTIMESTEP": tff / nsnap,
        "SOFTENING": softening,
        "GASSOFT": 2.0e-8,
        "TMAX": tff * tmax,
        "RHOMAX": ncrit,
        "BOXSIZE": boxsize,
        "OUTFOLDER": "output",
        "JET_PART_MASS": jet_particle_mass,
        "MS_WIND_PART_MASS": MS_wind_particle_mass,
        "BH_SEED_MASS": dm / 2.0,
        "TURBDECAY": tcross / 2,
        "TURBENERGY": turbenergy,
        "TURBFREQ": tcross / 20,
        "TURB_KMIN": int(100 * 2 * np.pi / L) / 100.0,
        "TURB_KMAX": int(100 * 4 * np.pi / (L) + 1) / 100.0,
        "TURB_SIGMA": (M_gas / 2e4) ** 0.5 * (R / 10) ** -0.5 * 600 / v_unit_SI * turbulence**0.5,
        "TURB_MINLAMBDA": int(100 * R / 2) / 100,
        "TURB_MAXLAMBDA": int(100 * R * 2) / 100,
        "TURB_COHERENCE_TIME": tcross / 2,
        "UNIT_L": 3.085678e18 * length_unit_pc,
        "UNIT_M": 1.989e33 * mass_unit_Msun,
        "UNIT_V": v_unit_SI * 1e2,
        "UNIT_B": B_unit_gauss,
        "ZINIT": metallicity,
        "ISRF": ISRF,
    }

    if not makebox:
        paramsfile = _read_params_template()
        for k, r in replacements.items():
            paramsfile = paramsfile.replace(k, (r if isinstance(r, str) else "{:.2e}".format(r)))
        open("params_" + filename.replace(".hdf5", "") + ".txt", "w").write(paramsfile)

    if makebox:
        box_side = float(Lbox) if Lbox is not None else 2 * R
        replacements_box = replacements.copy()
        replacements_box["NAME"] = filename.replace(".hdf5", "_BOX")
        replacements_box["BOXSIZE"] = box_side
        replacements_box["TURB_MINLAMBDA"] = int(100 * box_side / 2) / 100
        replacements_box["TURB_MAXLAMBDA"] = int(100 * box_side * 2) / 100
        paramsfile_box = _read_params_template()
        for k in replacements_box.keys():
            paramsfile_box = paramsfile_box.replace(k, str(replacements_box[k]))
        open("params_" + filename.replace(".hdf5", "") + "_BOX.txt", "w").write(paramsfile_box)

    if makecylinder:
        R_cyl = R * np.sqrt(np.pi / (4 * cyl_aspect_ratio))
        L_cyl = R_cyl * 2 * cyl_aspect_ratio
        vrms_cyl = (2 * G * M_gas / L_cyl) ** 0.5 * turbulence**0.5
        vrms_cyl *= 0.71
        tcross_cyl = 2 * R_cyl / vrms_cyl
        boxsize_cyl = L_cyl * 1.5 + R_cyl * 5
        print("Cylinder params: L=%g R=%g boxsize=%g vrms=%g" % (L_cyl, R_cyl, boxsize_cyl, vrms_cyl))
        replacements_cyl = replacements.copy()
        replacements_cyl["NAME"] = filename.replace(".hdf5", "_CYL")
        replacements_cyl["BOXSIZE"] = boxsize_cyl
        replacements_cyl["TURB_MINLAMBDA"] = int(100 * R_cyl) / 100
        replacements_cyl["TURB_MAXLAMBDA"] = int(100 * R_cyl * 4) / 100
        replacements_cyl["TURB_SIGMA"] = vrms_cyl
        replacements_cyl["TURB_COHERENCE_TIME"] = tcross_cyl / 2
        replacements_cyl["TURBDECAY"] = tcross_cyl / 2
        replacements_cyl["TURBENERGY"] = 0.019111097819633344 * vrms_cyl**3 / R_cyl
        replacements_cyl["TURBFREQ"] = tcross_cyl / 20
        replacements_cyl["TURB_KMIN"] = int(100 * 2 * np.pi / R_cyl) / 100.0
        replacements_cyl["TURB_KMAX"] = int(100 * 4 * np.pi / (R_cyl) + 1) / 100.0
        paramsfile_cyl = _read_params_template()
        for k in replacements_cyl.keys():
            paramsfile_cyl = paramsfile_cyl.replace(k, str(replacements_cyl[k]))
        open("params_" + filename.replace(".hdf5", "") + "_CYL.txt", "w").write(paramsfile_cyl)

    if param_only:
        print("Parameters only run, exiting...")
        return filename

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
    print("Done! Recomputing radii...")
    r = cdist(x, [np.zeros(3)])[:, 0]
    if np.any(r == 0):
        raise ValueError(
            "found point with r=0 in the glass file, we don't handle this case throughout our calculations yet. Stopping."
        )
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
        vt = TurbField(minmode=minmode, slope=turb_slope, sol_weight=turb_sol, seed=seed)
        nmin, nmax = vt.shape[-1] // 4, 3 * vt.shape[-1] // 4
        vt = vt[:, nmin:nmax, nmin:nmax, nmin:nmax]
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
    Eturb = 0.5 * dm * np.sum(v**2)
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
        3.429e8 * B_unit_gauss * (M_gas) ** -0.5 * R**1.5 * np.sqrt(4 * np.pi / 3) / v_unit_SI
    )
    uB = 0.5 * M_gas * vA_unit**2
    if B_fixed is not None:
        B_vec = np.atleast_1d(np.array(B_fixed, dtype=float))
        if B_vec.shape == (1,):
            B = np.c_[np.zeros(N_gas), np.zeros(N_gas), np.full(N_gas, B_vec[0])]
        elif B_vec.shape == (3,):
            B = np.tile(B_vec, (N_gas, 1))
        else:
            raise ValueError("B must be a scalar or 3-element vector")
    elif bfixed > 0:
        B = B * bfixed
    else:
        B = B * np.sqrt(magnetic_field * ugrav / uB)

    v = v - np.average(v, axis=0)
    x = x - np.average(x, axis=0)

    r, phi = np.sum(x**2, axis=1) ** 0.5, np.arctan2(x[:, 1], x[:, 0])
    theta = np.arccos(x[:, 2] / r)
    phi += phimode * np.sin(2 * phi) / 2
    x = r[:, np.newaxis] * np.c_[
        np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    ]

    if makecylinder:

        def ind_in_cylinder(x, L_cyl, R_cyl):
            return (np.abs(x[:, 0]) < L_cyl / 2) & (np.sum(x[:, 1:] ** 2, axis=1) < R_cyl**2)

        N_cyl = 0
        while N_cyl <= N_gas:
            x_cyl = np.random.rand(2 * N_gas, 3) * 2 - 1
            x_cyl[:, 0] *= L_cyl / 2
            x_cyl[:, 1] *= R_cyl
            x_cyl[:, 2] *= R_cyl
            x_cyl = x_cyl[ind_in_cylinder(x_cyl, L_cyl, R_cyl)]
            N_cyl = len(x_cyl)
        x_cyl = x_cyl[:N_gas]
        v_cyl = np.cross([1, 0, 0], x_cyl, axis=-1) / R_cyl
        v_cyl *= vrms_cyl

    u = np.ones_like(mgas) * 0.101 / 2.0

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

    u = np.ones_like(mgas) * (200 / v_unit_SI) ** 2

    if diffuse_gas:
        rho_warm = M_gas * 3 / (4 * np.pi * R**3) / density_contrast
        if derefinement:
            x0 = get_glass_coords(N_gas, glass_path)
            x0 = 2 * (x0 - 0.5)
            r0 = (x0 * x0).sum(1) ** 0.5
            x0, r0 = x0[r0.argsort()], r0[r0.argsort()]
            N_warm = int(4 * np.pi * rho_warm * (3 * R) ** 3 / 3 / dm)
            x_warm = x0[:N_warm] * 3 * R / r0[N_warm - 1]
            x0 = x0[N_warm:]
            r0 = r0[N_warm:]
            rnew = 3 * R * np.exp(np.arange(len(x0)) / N_warm / 3)
            x_warm = np.concatenate([x_warm, (rnew / r0)[:, None] * x0], axis=0)
            x_warm = x_warm[np.max(np.abs(x_warm), axis=1) < boxsize / 2]
            N_warm = len(x_warm)
            R_warm = (x_warm * x_warm).sum(1) ** 0.5
            mgas = np.concatenate([mgas, np.clip(dm * (R_warm / (3 * R)) ** 3, dm, np.inf)])
        else:
            M_warm = boxsize**3 * rho_warm
            N_warm = int(M_warm / dm)
            x_warm = get_glass_coords(N_warm, glass_path)[:N_warm]
            x_warm /= x_warm.max()
            x_warm = boxsize * x_warm - boxsize / 2
            if impact_dist == 0:
                x_warm = x_warm[np.sum(x_warm**2, axis=1) > R**2]
            N_warm = len(x_warm)
            mgas = np.concatenate([mgas, np.repeat(mgas.sum() / len(mgas), N_warm)])
        x = np.concatenate([x, x_warm])
        v = np.concatenate([v, np.zeros((N_warm, 3))])
        Bmag = np.average(np.sum(B**2, axis=1)) ** 0.5
        B = np.concatenate([B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([0, 0, 1])])
        u = np.concatenate([u, np.repeat(101.0, N_warm)])

        if makecylinder:
            B_cyl = np.concatenate([B, np.repeat(Bmag, N_warm)[:, np.newaxis] * np.array([1, 0, 0])])
            M_warm_cyl = (boxsize_cyl**3 - (4 * np.pi * R**3 / 3)) * rho_warm
            N_warm_cyl = int(M_warm_cyl / dm)
            x_warm_cyl = boxsize_cyl * np.random.rand(N_warm_cyl, 3) - boxsize_cyl / 2
            x_warm_cyl = x_warm_cyl[~ind_in_cylinder(x_warm_cyl, L_cyl, R_cyl)]
            N_warm_cyl = len(x_warm_cyl)
            x_cyl = np.concatenate([x_cyl, x_warm_cyl])
            v_cyl = np.concatenate([v_cyl, np.zeros((N_warm_cyl, 3))])
    else:
        N_warm = 0

    rho = np.repeat(3 * M_gas / (4 * np.pi * R**3), len(mgas))
    if diffuse_gas:
        rho[-N_warm:] /= 1000
    h = (32 * mgas / rho) ** (1.0 / 3)  # noqa: F841

    x += boxsize / 2
    if makecylinder:
        x_cyl += boxsize_cyl / 2

    print("Writing snapshot...")

    if not makebox:
        F = h5py.File(filename, "w")
        F.create_group("PartType0")
        F.create_group("Header")
        if star_list is not None:
            sl = np.atleast_2d(np.loadtxt(star_list))
            star_masses = sl[:, 0]
            star_ages = sl[:, 1]
            star_coords = sl[:, 2:5]
            star_vels = sl[:, 5:8] if sl.shape[1] >= 8 else np.zeros((len(star_masses), 3))
            N_stars = len(star_masses)
        else:
            N_stars = 1 if M_star > 0 else 0

        F["Header"].attrs["NumPart_ThisFile"] = [len(mgas), 0, 0, 0, 0, N_stars]
        F["Header"].attrs["NumPart_Total"] = [len(mgas), 0, 0, 0, 0, N_stars]
        F["Header"].attrs["BoxSize"] = boxsize
        F["Header"].attrs["Time"] = 0.0
        F["PartType0"].create_dataset("Masses", data=mgas)
        F["PartType0"].create_dataset("Coordinates", data=x)
        F["PartType0"].create_dataset("Velocities", data=v)
        F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(len(mgas)))
        F["PartType0"].create_dataset("InternalEnergy", data=u)

        if N_stars > 0:
            if star_list is not None:
                pt5_masses = star_masses
                pt5_ages = star_ages
                pt5_coords = star_coords
                pt5_vels = star_vels
            else:
                pt5_masses = np.array([M_star])
                pt5_ages = np.array([star_age])
                pt5_coords = np.array([x_star])
                pt5_vels = np.array([v_star])
            R_ZAMS = np.where(pt5_masses > 1.0, pt5_masses**0.57, pt5_masses**0.8)
            id_offset = F["PartType0/ParticleIDs"][:].max() + 1
            F.create_group("PartType5")
            F["PartType5"].create_dataset("Masses", data=pt5_masses)
            F["PartType5"].create_dataset("Coordinates", data=pt5_coords)
            F["PartType5"].create_dataset("Velocities", data=pt5_vels)
            F["PartType5"].create_dataset("ParticleIDs", data=id_offset + np.arange(N_stars))
            F["PartType5"].create_dataset("BH_Mass", data=pt5_masses)
            F["PartType5"].create_dataset("BH_Mass_AlphaDisk", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("BH_Mdot", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("BH_Specific_AngMom", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("SinkRadius", data=np.full(N_stars, softening))
            F["PartType5"].create_dataset("StellarFormationTime", data=-pt5_ages)
            F["PartType5"].create_dataset("ProtoStellarAge", data=pt5_ages)
            F["PartType5"].create_dataset(
                "ProtoStellarStage", data=np.full(N_stars, star_stage, dtype=np.int32), dtype=np.int32
            )
            F["PartType5"].create_dataset("ProtoStellarRadius_inSolar", data=R_ZAMS)
            F["PartType5"].create_dataset("StarLuminosity_Solar", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("Mass_D", data=np.zeros(N_stars))

        if magnetic_field > 0.0:
            F["PartType0"].create_dataset("MagneticField", data=B)
        F.close()

    if makebox:
        box_filename = filename.replace(".hdf5", "_BOX.hdf5")
        N_box = N_gas
        _cbrt = round(N_box ** (1 / 3))
        if _cbrt ** 3 == N_box and (_cbrt & (_cbrt - 1)) == 0:  # power of 8
            box_coords = _get_box_glass(N_box, os.path.expanduser("~") + "/.makecloud_glass") * box_side
        else:
            box_coords = np.random.rand(N_box, 3) * box_side
        box_mgas = np.repeat(dm, N_box)
        box_u = np.ones(N_box) * u[0]
        if star_list is not None:
            sl = np.atleast_2d(np.loadtxt(star_list))
            pt5_masses = sl[:, 0]
            pt5_ages = sl[:, 1]
            pt5_coords = sl[:, 2:5]
            pt5_vels = sl[:, 5:8] if sl.shape[1] >= 8 else np.zeros((len(sl), 3))
            N_stars = len(pt5_masses)
        elif M_star > 0:
            pt5_masses = np.array([M_star])
            pt5_ages = np.array([star_age])
            pt5_coords = np.array([x_star])
            pt5_vels = np.array([v_star])
            N_stars = 1
        else:
            N_stars = 0
        F = h5py.File(box_filename, "w")
        F.create_group("PartType0")
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [N_box, 0, 0, 0, 0, N_stars]
        F["Header"].attrs["NumPart_Total"] = [N_box, 0, 0, 0, 0, N_stars]
        F["Header"].attrs["MassTable"] = [M_gas / N_box, 0, 0, 0, 0, 0]
        F["Header"].attrs["BoxSize"] = box_side
        F["Header"].attrs["Time"] = 0.0
        F["PartType0"].create_dataset("Masses", data=box_mgas)
        F["PartType0"].create_dataset("Coordinates", data=box_coords)
        F["PartType0"].create_dataset("Velocities", data=np.zeros((N_box, 3)))
        F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(N_box))
        F["PartType0"].create_dataset("InternalEnergy", data=box_u)
        if N_stars > 0:
            R_ZAMS = np.where(pt5_masses > 1.0, pt5_masses**0.57, pt5_masses**0.8)
            F.create_group("PartType5")
            F["PartType5"].create_dataset("Masses", data=pt5_masses)
            F["PartType5"].create_dataset("Coordinates", data=pt5_coords)
            F["PartType5"].create_dataset("Velocities", data=pt5_vels)
            F["PartType5"].create_dataset("ParticleIDs", data=1 + N_box + np.arange(N_stars))
            F["PartType5"].create_dataset("BH_Mass", data=pt5_masses)
            F["PartType5"].create_dataset("BH_Mass_AlphaDisk", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("BH_Mdot", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("BH_Specific_AngMom", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("SinkRadius", data=np.full(N_stars, softening))
            F["PartType5"].create_dataset("StellarFormationTime", data=-pt5_ages)
            F["PartType5"].create_dataset("ProtoStellarAge", data=pt5_ages)
            F["PartType5"].create_dataset(
                "ProtoStellarStage", data=np.full(N_stars, star_stage, dtype=np.int32), dtype=np.int32
            )
            F["PartType5"].create_dataset("ProtoStellarRadius_inSolar", data=R_ZAMS)
            F["PartType5"].create_dataset("StarLuminosity_Solar", data=np.zeros(N_stars))
            F["PartType5"].create_dataset("Mass_D", data=np.zeros(N_stars))
        if magnetic_field > 0.0:
            F["PartType0"].create_dataset("MagneticField", data=B[:N_box])
        F.close()

    if makecylinder:
        cyl_filename = filename.replace(".hdf5", "_CYL.hdf5")
        F = h5py.File(cyl_filename, "w")
        F.create_group("PartType0")
        F.create_group("Header")
        F["Header"].attrs["NumPart_ThisFile"] = [N_gas + N_warm_cyl, 0, 0, 0, 0, 0]
        F["Header"].attrs["NumPart_Total"] = [N_gas + N_warm_cyl, 0, 0, 0, 0, 0]
        F["Header"].attrs["MassTable"] = [dm, 0, 0, 0, 0, 0]
        F["Header"].attrs["BoxSize"] = boxsize_cyl
        F["Header"].attrs["Time"] = 0.0
        F["PartType0"].create_dataset("Masses", data=mgas)
        F["PartType0"].create_dataset("Coordinates", data=x_cyl)
        F["PartType0"].create_dataset("Velocities", data=v_cyl)
        F["PartType0"].create_dataset("ParticleIDs", data=1 + np.arange(N_gas + N_warm_cyl))
        F["PartType0"].create_dataset("InternalEnergy", data=u)
        if magnetic_field > 0.0:
            F["PartType0"].create_dataset("MagneticField", data=B_cyl)
        F.close()

    return box_filename if makebox else filename
