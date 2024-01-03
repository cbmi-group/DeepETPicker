import numpy as np

def coords2star(data, save_path):
    """Relion star"""
    string = """
data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3"""

    data = np.round(data, 1).astype(str)

    with open(save_path, 'w') as f:
        f.writelines(string + '\n')
        for item in data:
            line = ' '.join(item)
            f.write(line + '\n')


def coords2box(data, save_path):
    """EMAN2 box"""
    with open(save_path, 'w') as f:
        for item in data:
            line = f"{item[0]:.1f}\t{item[1]:.1f}\t{item[2]:.0f}"
            f.write(line + '\n')

def coords2coords(data, save_path):
    """EMAN2 box"""
    with open(save_path, 'w') as f:
        for item in data:
            line = f"{item[0]:.0f}\t{item[1]:.0f}\t{item[2]:.0f}"
            f.write(line + '\n')
