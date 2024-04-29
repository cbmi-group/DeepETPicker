import warnings
import os
import glob
import numpy as np
import pandas as pd
import sys

warnings.simplefilter('ignore')
base_dir = "/ldap_shared/synology_shared/IBP_ribosome/liver_NewConstruct/Pick/reconstruction_2400"

def coords_gen(coord_path, coord_format, base_dir):
    os.makedirs(os.path.join(base_dir, 'coords'), exist_ok=True)

    coords_list = [i.split('/')[-1] for i in glob.glob(coord_path + f'/*{coord_format}')]
    coords_list = sorted(coords_list)

    num_all = []
    dir_names = []
    for dir in coords_list:
        data = []
        with open(os.path.join(coord_path, dir), 'r') as f:
            for idx, item in enumerate(f):
                data.append(item.rstrip('\n').split())
        try:
            data = np.array(data).astype(np.float).astype(int)
        except:
            data = np.array(data).astype(np.float32).astype(int)

        np.savetxt(os.path.join(base_dir, "coords", dir),
                   data, delimiter='\t', newline="\n", fmt="%s")
        num_all.append(data.shape[0])

        dir_name = dir[:-len(coord_format)]
        dir_names.append(dir_name)

    str_ = "|"
    for i in np.arange(0, len(num_all), 1):
        tmp = np.array(num_all)[:i+1].sum()
        print("0 to %d:" % (i+1), tmp)
        str_ += "%d|" % tmp
    # print(str_)
    # print(list(enumerate(num_all)))

    # gen num_name.csv
    num_name = np.array([num_all]).transpose()
    try:
        df = pd.DataFrame(num_name).astype(np.float)
    except:
        df = pd.DataFrame(num_name).astype(np.float32)
    df['dir_names'] = np.array([dir_names]).transpose()
    df['idx'] = np.arange(len(dir_names)).reshape(-1, 1)
    df.to_csv(os.path.join(base_dir, "coords", "num_name.csv"), sep='\t', header=False, index=False)


def coords_gen_show(args):
    coord_path, coord_format, base_dir, stdout = args
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    try:
        coords_gen(coord_path, coord_format, base_dir)
        print('Coord generation finished!')
        print('*' * 100)
    except:
        stdout.flush()
        stdout.write('Coordinates Generation Exception!')
        print('*' * 100)
        return 0

    if stdout is not None:
        sys.stderr = save_stderr
        sys.stdout = save_stdout