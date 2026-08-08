from astropy.io import fits

slx = r"E:\Projects\solar-flare-nowcast\data\aditya\solexs\pradan1.issdc.gov.in\al1\protected\downloadData\solexs\level1\2024\08\N00_0000\AL1_SLX_L1_20240802_v1.1\SDD2\AL1_SOLEXS_20240802_SDD2_L1.lc.gz"
hls = r"E:\Projects\solar-flare-nowcast\data\aditya\hel1os\pradan1.issdc.gov.in\al1\protected\downloadData\hel1os\level1\2024\08\02\N00_0000\2024\08\02\HLS_20240802_000011_43180sec_lev1_V111\cdte\lightcurve_cdte1.fits"

for f in (slx, hls):
    print("\n===", f.split("\\")[-1])
    with fits.open(f) as h:
        h.info()
        for i, hdu in enumerate(h):
            if hasattr(hdu, "columns"):
                print(f" HDU{i} cols:", [c.name for c in hdu.columns])
                d = hdu.data
                if d is not None and len(d):
                    print("   nrows:", len(d), "| first row:", d[0])