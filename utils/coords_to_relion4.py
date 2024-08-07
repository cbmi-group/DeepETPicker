from io import StringIO
import pandas as pd
import numpy as np
from pathlib import Path


def c2w(input_dir="coord_path/Coords_All",
        ouput_dir="./output",
        ouput_name="coords2relion4.star"):
    input_dir = Path(input_dir)
    ouput_dir = Path(ouput_dir)
    ouput_dir.mkdir(exist_ok=True, parents=True)
    ouput_path = ouput_dir / ouput_name
    coords_paths = sorted(list(input_dir.glob("*.txt")) + list(input_dir.glob("*.coords")))

    dfs = []
    for coords_path in coords_paths:
        coords_data = np.loadtxt(coords_path)
        XYZ = coords_data[:, -3:]

        TomoName = np.array([str(coords_path).split('/')[-1].split('.')[0]] * coords_data.shape[0], dtype=str).reshape(-1, 1)
        TomoParticleId = np.arange(1, coords_data.shape[0]+1).reshape(-1, 1)
        originXYZang = np.zeros_like(XYZ)
        angle = np.zeros_like(XYZ)
        if coords_data.shape[1] == 4:
            ClassNumber = coords_data[:, 0].reshape(-1, 1)
        else:
            ClassNumber = np.ones_like(TomoParticleId)
        randomsubset = np.array([1, 2] * (coords_data.shape[0]+1 // 2)).reshape(-1, 1)[:coords_data.shape[0]]
        df = pd.DataFrame(np.c_[TomoName, TomoParticleId,  XYZ.astype(np.int32), originXYZang, angle, ClassNumber.astype(np.int32), randomsubset])
        dfs.append(df)
    dfs = pd.concat(dfs, axis=0)

    with StringIO() as buffer:
        dfs.to_csv(buffer, sep="\t", index=False, header=None)
        lines = buffer.getvalue()

    with open(ouput_path, "w") as ofile:
        ofile.write("relion4" + "\n")
        ofile.write("data_particles" + "\n")
        ofile.write("loop_" + "\n")
        ofile.write("_rlnTomoName #1" + "\n")
        ofile.write("_rlnTomoParticleId #2" + "\n")
        ofile.write("_rlnCoordinateX #3" + "\n")
        ofile.write("_rlnCoordinateY #4" + "\n")
        ofile.write("_rlnCoordinateZ #5" + "\n")
        ofile.write("_rlnOriginXAngst #6" + "\n")
        ofile.write("_rlnOriginYAngst #7" + "\n")
        ofile.write("_rlnOriginZAngst #8" + "\n")
        ofile.write("_rlnAngleRot #9" + "\n")
        ofile.write("_rlnAngleTilt #10" + "\n")
        ofile.write("_rlnAnglePsi #11" + "\n")
        ofile.write("_rlnClassNumber #12" + "\n")
        ofile.write("_rlnRandomSubset #13" +"\n")
        ofile.write(lines)
    print(f"Save: {ouput_path}")


if __name__ == "__main__":
    c2w()
