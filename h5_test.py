import h5py

fp = "/data/Data_yuq/FY4A/20190105/FY4A-_AGRI--_N_DISK_1047E_L1-_GEO-_MULT_NOM_20190105040000_20190105041459_4000M_V0001.HDF"

with h5py.File(fp, "r") as f:
    print("ROOT ATTRS:")
    for k in [
        "NOMCenterLat", "NOMCenterLon", "NOMSatHeight",
        "RegCenterLat", "RegCenterLon", "RegLength", "RegWidth",
        "dEA", "dObRecFlat", "dSamplingAngle", "dSteppingAngle"
    ]:
        if k in f.attrs:
            print(k, f.attrs[k])

    print("\nColumnNumber attrs:")
    for k, v in f["ColumnNumber"].attrs.items():
        print(k, v)

    print("\nLineNumber attrs:")
    for k, v in f["LineNumber"].attrs.items():
        print(k, v)