import mrcfile
from multiprocessing import Pool
import pandas as pd
import os
import numpy as np
from glob import glob
import sys

def gaussian3D(shape, sigma=1):
    l, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-l:l + 1, -m:m + 1, -n:n + 1]
    sigma = (sigma - 1.) / 2.
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    # h[h < np.finfo(float).eps * h.max()] = 0
    return h


class Coord_to_Label():
    def __init__(self, base_path, coord_path, coord_format, tomo_path, tomo_format,
                 num_cls, label_type, label_diameter):

        self.base_path = base_path
        self.coord_path = coord_path
        self.coord_format = coord_format
        self.tomo_path = tomo_path
        self.tomo_format = tomo_format
        self.num_cls = num_cls
        self.label_type = label_type
        self.label_diameter = [int(i) for i in label_diameter.split(',')]

        if 'ocp' in self.label_type.lower():
            self.label_path = os.path.join(self.base_path, self.label_type)
        else:
            self.label_path = os.path.join(self.base_path,
                                           self.label_type + str(self.label_diameter[0]))
        os.makedirs(self.label_path, exist_ok=True)

        self.dir_list = [i[:-len(self.coord_format)] for i in os.listdir(self.coord_path) if self.coord_format in i]
        self.names = [i + self.tomo_format for i in self.dir_list]  # if self.tomo_format not in i else i

    def single_handle(self, i):
        self.tomo_file = f"{self.tomo_path}/{self.names[i]}"
        data_file = mrcfile.open(self.tomo_file, permissive=True)
        print(os.path.join(self.label_path, self.names[i]))
        label_file = mrcfile.new(os.path.join(self.label_path, self.names[i]),
                                 overwrite=True)

        label_positions = pd.read_csv(os.path.join(self.base_path, 'coords', '%s.coords' % self.dir_list[i]), sep='\t',
                                      header=None).to_numpy()

        # template = np.fromfunction(lambda i, j, k: (i - r) * (i - r) + (j - r) * (j - r) + (k - r) * (k - r) <= r * r,
        #                            (2 * r + 1, 2 * r + 1, 2 * r + 1), dtype=int).astype(int)

        z_max, y_max, x_max = data_file.data.shape
        try:
            label_data = np.zeros(data_file.data.shape, dtype=np.float)
        except:
            label_data = np.zeros(data_file.data.shape, dtype=np.float32)

        for pos_idx, a_pos in enumerate(label_positions):
            if self.num_cls == 1 and len(a_pos) == 3:
                x, y, z = a_pos
                cls_idx_ = 1
            else:
                cls_idx_, x, y, z = a_pos

            if 'data_ocp' in self.label_type.lower():
                dim = int(self.label_diameter[cls_idx_ - 1])
            else:
                dim = int(self.label_diameter[0])
            radius = int(dim / 2)
            r = radius

            template = gaussian3D((dim, dim, dim), dim)

            cls_idx = pos_idx+1 if 'data_ocp' in self.label_type else cls_idx_
            # print(self.label_type, dim, cls_idx)
            z_start = 0 if z - r < 0 else z - r
            z_end = z_max if z + r + 1 > z_max else z + r + 1
            y_start = 0 if y - r < 0 else y - r
            y_end = y_max if y + r + 1 > y_max else y + r + 1
            x_start = 0 if x - r < 0 else x - r
            x_end = x_max if x + r + 1 > x_max else x + r + 1

            t_z_start = r - z if z - r < 0 else 0
            t_z_end = (r + z_max - z) if z + r + 1 > z_max else 2 * r + 1
            t_y_start = r - y if y - r < 0 else 0
            t_y_end = (r + y_max - y) if y + r + 1 > y_max else 2 * r + 1
            t_x_start = r - x if x - r < 0 else 0
            t_x_end = (r + x_max - x) if x + r + 1 > x_max else 2 * r + 1

            # print(z_start, z_end, y_start, y_end, x_start, x_end)
            # check border
            # print(label_data.shape)
            # print(z_start, z_end, y_start, y_end, x_start, x_end)
            tmp1 = label_data[z_start:z_end, y_start:y_end, x_start:x_end]
            tmp2 = template[t_z_start:t_z_end, t_y_start:t_y_end, t_x_start:t_x_end]

            larger_index = tmp1 < tmp2
            tmp1[larger_index] = tmp2[larger_index]

            if 'cubic' in self.label_type.lower():
                tg = 0.223  # exp(-1.5)
            elif 'sphere' in self.label_type.lower() or 'ocp' in self.label_type.lower():
                tg = 0.60653 # exp(-0.5)
            else:
                tg = 0.367879 # exp(-1)
            tmp1[tmp1 <= tg] = 0
            tmp1 = np.where(tmp1 > 0, cls_idx, 0)
            label_data[z_start:z_end, y_start:y_end, x_start:x_end] = tmp1

        label_file.set_data(label_data)

        data_file.close()
        label_file.close()
        print('work %s done' % i)
        # return 'work %s done' % i

    def gen_labels(self):
        if len(self.dir_list) == 1:
            self.single_handle(0)
        else:
            with Pool(len(self.dir_list)) as p:
                p.map(self.single_handle, np.arange(len(self.dir_list)).tolist())


def label_gen_show(args):
    base_path, coord_path, coord_format, tomo_path, tomo_format, \
    num_cls, label_type, label_diameter, stdout = args
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    try:
        label_gen = Coord_to_Label(base_path, coord_path, coord_format, tomo_path, tomo_format, \
                    num_cls, label_type, label_diameter)
        label_gen.gen_labels()
        if 'ocp' not in label_type:
            print('Label generation finished!')
        else:
            print('Occupancy generation finished!')
    except:
        stdout.flush()
        stdout.write('Label Generation Exception!')
        return 0

    sys.stderr = save_stderr
    sys.stdout = save_stdout


class Coord_to_Label_v1():
    def __init__(self, tomo_file, coord_file, num_cls, label_diameter, label_type):

        self.tomo_file = tomo_file
        self.coord_file = coord_file
        self.num_cls = num_cls
        self.label_type = label_type
        self.label_diameter = label_diameter

    def gen_labels(self):
        if '.coords' in self.coord_file or '.txt' in self.coord_file:
            data_file = mrcfile.open(self.tomo_file, permissive=True)

            label_positions = pd.read_csv(self.coord_file, sep='\t', header=None).to_numpy()
            if self.label_type == 'Coords':
                return label_positions

            dim = int(self.label_diameter)
            radius = int(dim / 2)
            r = radius

            template = gaussian3D((dim, dim, dim), dim)

            z_max, y_max, x_max = data_file.data.shape
            try:
                label_data = np.zeros(data_file.data.shape, dtype=np.float)
            except:
                label_data = np.zeros(data_file.data.shape, dtype=np.float32)

            for pos_idx, a_pos in enumerate(label_positions):
                if self.num_cls == 1 and len(a_pos) == 3:
                    x, y, z = a_pos
                    cls_idx = 1
                else:
                    cls_idx, x, y, z = a_pos
                cls_idx = pos_idx+1 if 'ocp' in self.label_type else cls_idx
                z_start = 0 if z - r < 0 else z - r
                z_end = z_max if z + r + 1 > z_max else z + r + 1
                y_start = 0 if y - r < 0 else y - r
                y_end = y_max if y + r + 1 > y_max else y + r + 1
                x_start = 0 if x - r < 0 else x - r
                x_end = x_max if x + r + 1 > x_max else x + r + 1

                t_z_start = r - z if z - r < 0 else 0
                t_z_end = (r + z_max - z) if z + r + 1 > z_max else 2 * r + 1
                t_y_start = r - y if y - r < 0 else 0
                t_y_end = (r + y_max - y) if y + r + 1 > y_max else 2 * r + 1
                t_x_start = r - x if x - r < 0 else 0
                t_x_end = (r + x_max - x) if x + r + 1 > x_max else 2 * r + 1

                # print(z_start, z_end, y_start, y_end, x_start, x_end)
                # check border
                tmp1 = label_data[z_start:z_end, y_start:y_end, x_start:x_end]
                tmp2 = template[t_z_start:t_z_end, t_y_start:t_y_end, t_x_start:t_x_end]

                larger_index = tmp1 < tmp2
                tmp1[larger_index] = tmp2[larger_index]
                tmp1[tmp1 <= 0.36788] = 0

                tmp1 = np.where(tmp1 > 0, cls_idx, 0)

                label_data[z_start:z_end, y_start:y_end, x_start:x_end] = tmp1

            data_file.close()
            return label_data
        elif '.mrc' in self.tomo_file or '.rec' in self.tomo_file:
            label_data = mrcfile.open(self.coord_file, permissive=True)
            return label_data.data

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from configs.c2l_10045_New_bin8_mask3 import pre_config
    from utils.coord_gen import coords_gen
    from utils.normalization import InputNorm

    # 初始化坐标文件为整数
    coords_gen(pre_config["coord_path"],
               pre_config["base_path"])

    # 归一化
    pre_norm = InputNorm(pre_config["tomo_path"],
               pre_config["tomo_format"],
               pre_config["base_path"],
               pre_config["norm_type"])
    pre_norm.handle_parallel()

    # 根据coords产生labels
    c2l = Coord_to_Label(base_path=pre_config["base_path"],
                         coord_path=pre_config["coord_path"],
                         coord_format=pre_config["coord_format"],
                         tomo_path=pre_config["tomo_path"],
                         tomo_format=pre_config["tomo_format"],
                         num_cls=pre_config["num_cls"],
                         label_type=pre_config["label_type"],
                         label_diameter=pre_config["label_diameter"],
                         )
    c2l.gen_labels()

    # 根据coords产生ocps
    if pre_config["label_diameter"] !=pre_config["ocp_diameter"]:
        c2l = Coord_to_Label(base_path=pre_config["base_path"],
                             coord_path=pre_config["coord_path"],
                             coord_format=pre_config["coord_format"],
                             tomo_path=pre_config["tomo_path"],
                             tomo_format=pre_config["tomo_format"],
                             num_cls=pre_config["num_cls"],
                             label_type=pre_config["ocp_type"],
                             label_diameter=pre_config["ocp_diameter"],
                             )
        c2l.gen_labels()
