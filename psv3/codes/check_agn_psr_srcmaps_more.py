from astropy.io import fits

agn = "agn_300M_1G_z100_srcmpas.fits"
psr = "psr_1_300M_1G_z100_srcmpas.fits"

with fits.open(agn) as ha, fits.open(psr) as hp:
    ka = ha[0].header
    kp = hp[0].header

    keys = sorted(set(ka.keys()) | set(kp.keys()))

    for key in keys:
        va = ka.get(key, "<MISSING>")
        vp = kp.get(key, "<MISSING>")

        if va != vp:
            print(f"{key:12s}  AGN={va!r}   PSR={vp!r}")
