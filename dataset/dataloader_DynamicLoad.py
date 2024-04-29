import os
import torch.utils.data as data
import numpy as np
import mrcfile
import pandas as pd
import torch
import warnings
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from torch.utils.data import DataLoader


class Dataset_ClsBased(data.Dataset):
    def __init__(self,
                 mode='train',
                 block_size=72,
                 num_class=1,
                 random_num=0,
                 use_bg=True,
                 data_split=[7, 7, 7],
                 test_use_pad=False,
                 pad_size=18,
                 use_paf=False,
                 cfg=None,
                 args=None):

        self.args = args
        self.mode = mode
        # use_CL CL = cnotrastive learning
        self.use_CL = args.use_CL
        if args.use_CL:
            self.radius = 0
        else:
            self.radius = block_size // 5
        self.use_bg = use_bg
        self.use_paf = use_paf
        self.use_CL_DA = args.use_CL_DA
        self.use_bg_part = args.use_bg_part
        self.use_ice_part = args.use_ice_part
        self.Sel_Referance = args.Sel_Referance

        pad_size = pad_size[0] if isinstance(pad_size, list) else pad_size
        base_dir = cfg['base_path']
        label_name = cfg['label_name'],
        coord_format = cfg['coord_format']
        tomo_format = cfg['tomo_format'],
        norm_type = cfg['norm_type']

        base_dir = base_dir[0] if isinstance(base_dir, tuple) else base_dir
        label_name = label_name[0] if isinstance(label_name, tuple) else label_name
        coord_format = coord_format[0] if isinstance(coord_format, tuple) else coord_format
        tomo_format = tomo_format[0] if isinstance(tomo_format, tuple) else tomo_format
        norm_type = norm_type[0] if isinstance(norm_type, tuple) else norm_type

        if 'label_path' not in cfg.keys():
            label_path = os.path.join(base_dir, label_name)
        else:
            label_path = cfg['label_path']
            label_path = label_path[0] if isinstance(label_path, tuple) else label_path

        if 'coord_path' not in cfg.keys():
            coord_path = os.path.join(base_dir, 'coords')
        else:
            coord_path = cfg['coord_path']
            coord_path = coord_path[0] if isinstance(coord_path, tuple) else coord_path

        if 'tomo_path' not in cfg.keys():
            if norm_type == 'standardization':
                tomo_path = base_dir + '/data_std'
            elif norm_type == 'normalization':
                tomo_path = base_dir + '/data_norm'
        else:
            tomo_path = cfg['tomo_path']
            tomo_path = tomo_path[0] if isinstance(tomo_path, tuple) else tomo_path

        ocp_name = cfg["ocp_name"]

        if 'ocp_path' not in cfg.keys():
            ocp_path = os.path.join(base_dir, ocp_name)
        else:
            ocp_path = cfg['ocp_path']
            ocp_path = ocp_path[0] if isinstance(ocp_path, tuple) else ocp_path

        print('*' * 100)
        print('num_name:', os.path.join(coord_path, 'num_name.csv'))
        print(f'base_path:{base_dir}')
        print(f"coord_path:{coord_path}")
        print(f"tomo_path:{tomo_path}")
        print(f"label_path:{label_path}")
        print(f"ocp_path:{ocp_path}")
        if self.use_paf:
            print(f"paf_path:{cfg['paf_path']}")
        print(f"label_name:{label_name}")
        print(f"coord_format:{coord_format}")
        print(f"tomo_format:{tomo_format}")
        print(f"norm_type:{norm_type}")
        print(f"ocp_name:{ocp_name}")
        print('*' * 100)

        if self.mode == 'test_only':
            num_name = pd.read_csv(os.path.join(tomo_path, 'num_name.csv'), sep='\t', header=None)
        else:
            num_name = pd.read_csv(os.path.join(coord_path, 'num_name.csv'), sep='\t', header=None)

        dir_names = num_name.iloc[:, 1].to_numpy().tolist()
        print(num_name)
        # print(dir_names)

        if self.mode == 'train':
            self.data_range = np.arange(data_split[0], data_split[1])
        elif self.mode == 'val':
            self.data_range = np.arange(data_split[2], data_split[3])
        else:  # test or test_val or val_v1
            self.data_range = np.arange(data_split[4], data_split[5])
        print(f"data_range:{self.data_range}")

        # print(f'data_range:{self.data_range}')
        self.shift = block_size // 2  # bigger than self.radius to cover full particle
        self.num_class = num_class

        self.ground_truth_volume = []
        self.class_mask = []
        self.location = []
        self.origin = []
        self.label = []

        # inital Data Augmentation
        if self.use_bg and self.mode == 'train':
            patch_size = [block_size] * 3
            self.st = SpatialTransform_2(
                patch_size, [i // 2 for i in patch_size],
                do_elastic_deform=True, deformation_scale=(0, 0.05),
                do_rotation=True,
                angle_x=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_y=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                angle_z=(- 15 / 360. * 2 * np.pi, 15 / 360. * 2 * np.pi),
                do_scale=True, scale=(0.95, 1.05),
                border_mode_data='constant', border_cval_data=0,
                border_mode_seg='constant', border_cval_seg=0,
                order_seg=0, order_data=3,
                random_crop=True,
                p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1
            )

            self.mt = MirrorTransform(axes=(0, 1, 2))
        # to avoid mrcfile warnings
        warnings.simplefilter('ignore')

        if self.mode == 'train' or self.mode == 'test_val' or self.mode == 'val':
            self.position = [
                pd.read_csv(os.path.join(coord_path, dir_names[i] + coord_format),
                            sep='\t', header=None).to_numpy() for i in self.data_range]

        # load Tomo
        if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'val_v1' \
                or self.mode == 'test_only':
            self.tomo_shape = []
            for idx in self.data_range:
                if not args.input_cat:
                    with mrcfile.open(os.path.join(tomo_path, dir_names[idx] + tomo_format),
                                      permissive=True) as gm:
                        shape = gm.data.shape
                        self.data_shape = gm.data.shape
                        shape_pad = [i + 2 * pad_size for i in shape]
                        try:
                            temp = np.zeros(shape_pad).astype(np.float)
                        except:
                            temp = np.zeros(shape_pad).astype(np.float32)
                        temp[pad_size:shape_pad[0] - pad_size,
                        pad_size:shape_pad[1] - pad_size,
                        pad_size:shape_pad[2] - pad_size] = gm.data
                        self.origin.append(temp)
                        self.tomo_shape.append(temp.shape)
                else:
                    for idx, p_suffix in enumerate(args.input_cat_items):
                        p_suffix = p_suffix.rstrip(',')
                        p_suffix = '' if p_suffix == 'None' else p_suffix
                        with mrcfile.open(
                                os.path.join(tomo_path + p_suffix, dir_names[self.data_range[0]] + tomo_format), permissive=True) as tmp:
                            if idx == 0:
                                gm = np.array(tmp.data)[None, ...]
                            else:
                                gm = np.concatenate([gm, np.array(tmp.data)[None, ...]], axis=0)

                    shape = gm.shape
                    self.data_shape = gm.shape
                    shape_pad = [shape[0]]
                    shape_pad.extend([i + 2 * pad_size for i in shape[1:]])
                    try:
                        temp = np.zeros(shape_pad).astype(np.float)
                    except:
                        temp = np.zeros(shape_pad).astype(np.float32)
                    temp[:, pad_size:shape_pad[1] - pad_size,
                    pad_size:shape_pad[2] - pad_size,
                    pad_size:shape_pad[3] - pad_size] = gm
                    self.origin.append(temp)
                    self.tomo_shape.append(temp.shape)

        else:
            if not args.input_cat:
                print([os.path.join(tomo_path, dir_names[i] + tomo_format) for i in self.data_range])
                self.origin = [mrcfile.open(os.path.join(tomo_path, dir_names[i] + tomo_format)) for i
                           in self.data_range]
            else:
                for idx, p_suffix in enumerate(args.input_cat_items):
                    p_suffix = p_suffix.rstrip(',')
                    p_suffix = '' if p_suffix == 'None' else p_suffix
                    with mrcfile.open(os.path.join(tomo_path + p_suffix, dir_names[self.data_range[0]] + tomo_format), permissive=True) as tmp:
                        if idx == 0:
                            self.origin = np.array(tmp.data)[None, ...]
                        else:
                            self.origin = np.concatenate([self.origin, np.array(tmp.data)[None, ...]], axis=0)

        # load Labels
        if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'val_v1':
            for idx in self.data_range:
                if os.path.exists(os.path.join(label_path, dir_names[idx] + tomo_format)):
                    with mrcfile.open(os.path.join(label_path, dir_names[idx] + tomo_format),
                                      permissive=True) as cm:
                        shape = cm.data.shape
                        shape_pad = [i + 2 * pad_size if i > pad_size else i for i in shape]
                        try:
                            temp = np.zeros(shape_pad).astype(np.float)
                        except:
                            temp = np.zeros(shape_pad).astype(np.float32)

                        if len(shape) == 3:
                            temp[pad_size:shape_pad[-3] - pad_size,
                            pad_size:shape_pad[-2] - pad_size,
                            pad_size:shape_pad[-1] - pad_size] = cm.data
                        elif len(shape) == 4:
                            print(temp.shape)
                            temp[:, pad_size:shape_pad[-3] - pad_size,
                            pad_size:shape_pad[-2] - pad_size,
                            pad_size:shape_pad[-1] - pad_size] = cm.data
                        self.label.append(temp)
                elif self.mode == 'test_val' and args.use_cluster:
                    self.label = [
                        np.zeros_like(self.origin[idx]) for idx, _ in enumerate(self.data_range)]
        elif self.mode == 'test_only':
            self.label = [
                np.zeros_like(self.origin[idx]) for idx, _ in enumerate(self.data_range)]
        else:
            self.label = [
                mrcfile.open(os.path.join(label_path, dir_names[idx] + tomo_format)) for idx
                in self.data_range]

        # load paf
        if self.use_paf:
            paf_path = cfg["paf_path"]
            paf_path = paf_path[0] if isinstance(paf_path, tuple) else paf_path

            self.paf_label = []
            if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'val_v1':
                for idx in self.data_range:
                    with mrcfile.open(os.path.join(paf_path, dir_names[idx] + tomo_format),
                                      permissive=True) as cm:
                        shape = cm.data.shape
                        shape_pad = [i + 2 * pad_size for i in shape]
                        try:
                            temp = np.zeros(shape_pad).astype(np.float)
                        except:
                            temp = np.zeros(shape_pad).astype(np.float32)
                        temp[pad_size:shape_pad[0] - pad_size,
                        pad_size:shape_pad[1] - pad_size,
                        pad_size:shape_pad[2] - pad_size] = cm.data
                        self.paf_label.append(temp)
            else:
                self.paf_label = [
                    mrcfile.open(os.path.join(paf_path, dir_names[idx] + tomo_format)) for idx
                    in self.data_range]

        # Generate BlockData
        self.coords = []
        self.data = []
        if self.mode == 'train' or self.mode == 'val':
            for i in range(len(self.data_range)):
                for j, point1 in enumerate(self.position[i]):
                    # if sel_train_num > 0 and j >= sel_train_num:
                    #     continue
                    if args.Sel_Referance:
                        if j in args.sel_train_num:
                            self.coords.append([i, point1[-3], point1[-2], point1[-1]])
                    else:
                        if point1[0] == 15:
                            for _ in range(13):
                                self.coords.append([i, point1[-3], point1[-2], point1[-1]])
                        else:
                            self.coords.append([i, point1[-3], point1[-2], point1[-1]])
        else:
            if test_use_pad:
                step_size = block_size - 2 * pad_size
            else:
                step_size = int(self.shift * 2)
            print(self.shift, step_size)
            for i in range(len(self.data_range)):
                shape = self.origin[i].shape[-3:]
                for j in range((shape[0] - 2 * pad_size) // step_size + (
                        1 if (shape[0] - 2 * pad_size) % step_size > 0 else 0)):
                    for k in range((shape[1] - 2 * pad_size) // step_size + (
                            1 if (shape[1] - 2 * pad_size) % step_size > 0 else 0)):
                        for l in range((shape[2] - 2 * pad_size) // step_size + (
                                1 if (shape[2] - 2 * pad_size) % step_size > 0 else 0)):
                            if j == (shape[0] - 2 * pad_size) // step_size + (
                                    1 if (shape[0] - 2 * pad_size) % step_size > 0 else 0) - 1:
                                z = shape[0] - block_size // 2
                            else:
                                z = j * step_size + block_size // 2

                            if k == (shape[1] - 2 * pad_size) // step_size + (
                                    1 if (shape[1] - 2 * pad_size) % step_size > 0 else 0) - 1:
                                y = shape[1] - block_size // 2
                            else:
                                y = k * step_size + block_size // 2

                            if l == (shape[2] - 2 * pad_size) // step_size + (
                                    1 if (shape[2] - 2 * pad_size) % step_size > 0 else 0) - 1:
                                x = shape[2] - block_size // 2
                            else:
                                x = l * step_size + block_size // 2
                            self.coords.append([i, x, y, z])

                            if len(self.origin[i].shape) == 4:
                                img = self.origin[i][:, z - self.shift: z + self.shift,
                                      y - self.shift: y + self.shift,
                                      x - self.shift: x + self.shift]
                            else:
                                img = self.origin[i][z - self.shift: z + self.shift,
                                      y - self.shift: y + self.shift,
                                      x - self.shift: x + self.shift]

                            if len(self.label[i].shape) == 4:
                                lab = self.label[i][:, z - self.shift: z + self.shift,
                                      y - self.shift: y + self.shift,
                                      x - self.shift: x + self.shift]
                            else:
                                lab = self.label[i][z - self.shift: z + self.shift,
                                      y - self.shift: y + self.shift,
                                      x - self.shift: x + self.shift]

                            if self.use_paf:
                                paf = self.paf_label[i][z - self.shift: z + self.shift,
                                      y - self.shift: y + self.shift,
                                      x - self.shift: x + self.shift]
                                self.data.append([img, lab, paf, [z, y, x]])
                            else:
                                self.data.append([img, lab, [z, y, x]])

        # add random samples
        if self.mode == 'train' and random_num > 0:
            print('random samples num:', random_num)
            for j in range(random_num):
                i = np.random.randint(len(self.data_range))
                data_shape = self.origin[i].data.shape
                z = np.random.randint(self.shift + 1, data_shape[0] - self.shift)
                y = np.random.randint(self.shift + 1, data_shape[1] - self.shift)
                x = np.random.randint(self.shift + 1, data_shape[2] - self.shift)
                self.coords.append([i, x, y, z])

        if self.mode == 'train':
            print("Training dataset contains {} samples".format((len(self.coords))))
        if self.mode == 'val':
            print("Validation dataset contains {} samples".format((len(self.coords))))
        if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'val_v1' or self.mode == 'test_only':
            print("Test dataset contains {} samples".format((len(self.coords))))
        self.test_len = len(self.coords)

        if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'test_only':
            self.dir_name = dir_names[self.data_range[0]]
        if self.mode == 'test' or self.mode == 'test_val':
            print(os.path.join(ocp_path, dir_names[self.data_range[0]] + tomo_format))
            if os.path.exists(os.path.join(ocp_path, dir_names[self.data_range[0]] + tomo_format)):
                with mrcfile.open(os.path.join(ocp_path, dir_names[self.data_range[0]] + tomo_format), permissive=True) as f:
                    self.occupancy_map = f.data
            elif self.mode == 'test_val' and args.use_cluster:
                self.occupancy_map = np.zeros_like(self.origin[0])
            self.gt_coords = pd.read_csv(os.path.join(coord_path, "%s.coords" % dir_names[self.data_range[0]]),
                     sep='\t', header=None).to_numpy()

        if args.use_bg_part and self.Sel_Referance:
            self.coords_bg = pd.read_csv(os.path.join(coord_path, dir_names[self.data_range[0]] + '_bg' + coord_format),
                                         sep='\t', header=None).to_numpy()[:len(self.coords)]
        if args.use_ice_part and self.Sel_Referance:
            self.coords_ice = pd.read_csv(os.path.join(coord_path, dir_names[self.data_range[0]] + '_ice' + coord_format),
                                         sep='\t', header=None).to_numpy()[:len(self.coords)]

        if self.mode == 'test_val':
            for i in range(len(self.data_range)):
                for j, point1 in enumerate(self.position[i]):
                    x, y, z = point1[-3] + pad_size, point1[-2] + pad_size, point1[-1] + pad_size
                    z_max, y_max, x_max = self.origin[i].data.shape
                    x, y, z = self.__sample(np.array([x, y, z]),
                                          np.array([x_max, y_max, z_max]))
                    img = self.origin[i][z - self.shift: z + self.shift,
                          y - self.shift: y + self.shift,
                          x - self.shift: x + self.shift]
                    if len(self.label[i].shape) == 4:
                        lab = self.label[i][:, z - self.shift: z + self.shift,
                              y - self.shift: y + self.shift,
                              x - self.shift: x + self.shift]
                    else:
                        lab = self.label[i][z - self.shift: z + self.shift,
                              y - self.shift: y + self.shift,
                              x - self.shift: x + self.shift]
                    if self.use_paf:
                        paf = self.paf_label[i][z - self.shift: z + self.shift,
                              y - self.shift: y + self.shift,
                              x - self.shift: x + self.shift]
                        self.data.append([img, lab, paf, [z, y, x]])
                    else:
                        self.data.append([img, lab, [z, y, x]])
        if self.mode == 'test_val' and args.use_cluster:
            self.data = self.data[-len(self.position[0]):]

    def __getitem__(self, index):
        if self.mode == 'test' or self.mode == 'test_val' or self.mode == 'val_v1' or self.mode =='test_only':
            if self.use_paf:
                img, label, paf_label, position = self.data[index]
            else:
                img, label, position = self.data[index]

        else:
            idx, x, y, z = self.coords[index]
            z_max, y_max, x_max = self.origin[idx].data.shape

            point = self.__sample(np.array([x, y, z]),
                                  np.array([x_max, y_max, z_max]))

            if self.args.input_cat:
                img = self.origin[:, point[2] - self.shift:point[2] + self.shift,
                      point[1] - self.shift:point[1] + self.shift,
                      point[0] - self.shift:point[0] + self.shift]
            else:
                img = self.origin[idx].data[point[2] - self.shift:point[2] + self.shift,
                      point[1] - self.shift:point[1] + self.shift,
                      point[0] - self.shift:point[0] + self.shift]

            if len(self.label[idx].data.shape) == 4:
                label = self.label[idx].data[:, point[2] - self.shift:point[2] + self.shift,
                        point[1] - self.shift:point[1] + self.shift,
                        point[0] - self.shift:point[0] + self.shift]
            else:
                label = self.label[idx].data[point[2] - self.shift:point[2] + self.shift,
                        point[1] - self.shift:point[1] + self.shift,
                        point[0] - self.shift:point[0] + self.shift]
            position = [point[2], point[1], point[0]]
            if self.use_paf:
                paf_label = self.paf_label[idx].data[point[2] - self.shift:point[2] + self.shift,
                        point[1] - self.shift:point[1] + self.shift,
                        point[0] - self.shift:point[0] + self.shift]
        # print(img.shape, label.shape)
        img = np.array(img)
        try:
            label = np.array(label).astype(np.float)
        except:
            label = np.array(label).astype(np.float32)

        if self.num_class > 1 and len(label.shape) == 3:
            label = multiclass_label(label,
                                     num_classes=self.num_class,
                                     first_idx=1 if self.use_paf else 0)
        else:
            if self.mode == 'test' and label.shape != (self.shift * 2, self.shift * 2, self.shift * 2):
                label = np.zeros((1, self.shift * 2, self.shift * 2, self.shift * 2))
            else:
                label = label.reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)

        if self.use_paf:
            try:
                paf_label = np.array(paf_label).astype(np.float).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
            except:
                paf_label = np.array(paf_label).astype(np.float32).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)

            label = np.concatenate([label, paf_label], axis=0)

        if self.use_CL_DA:
            img = self.__DA_SelReference(img)

        # random 3D rotation
        if self.mode == 'train' and not self.use_CL:
            if self.use_bg:
                img_label = {'data': img.reshape(1, -1, self.shift * 2, self.shift * 2, self.shift * 2),
                             'seg': label.reshape(1, -1, self.shift * 2, self.shift * 2, self.shift * 2)}
                if torch.rand(1) < 0.5:
                    img_label = self.st(**img_label)
                else:
                    img_label = self.mt(**img_label)
                img = img_label['data'].reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
                label = img_label['seg'].reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
            else:
                img = np.array(img).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
                label = np.array(label).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
                # degree = np.random.randint(4, size=3)
                # img = self.__rotation3D(img, degree)
                # label = self.__rotation3D(label, degree)
                # img = np.array(img)
                # label = np.array(label)
        else:
            img = img.reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
            label = label.reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)

        img = torch.as_tensor(img).float()
        label = torch.as_tensor(label).float()

        if self.use_bg_part and self.Sel_Referance:
            idx, x, y, z = self.coords_bg[index]
            z_max, y_max, x_max = self.origin[0].data.shape

            # point = self.__sample(np.array([x, y, z]),
            #                       np.array([x_max, y_max, z_max]))
            point = [x, y, z]

            img_bg = self.origin[0].data[point[2] - self.shift:point[2] + self.shift,
                  point[1] - self.shift:point[1] + self.shift,
                  point[0] - self.shift:point[0] + self.shift]

            if self.use_CL_DA:
                img_bg = self.__DA_SelReference(np.array(img_bg))

            img_bg = np.array(img_bg).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
            img_bg = torch.as_tensor(img_bg).float()

        if self.use_ice_part and self.Sel_Referance:
            idx, x, y, z = self.coords_ice[index]
            z_max, y_max, x_max = self.origin[0].data.shape

            # point = self.__sample(np.array([x, y, z]),
            #                       np.array([x_max, y_max, z_max]))
            point = [x, y, z]
            img_ice = self.origin[0].data[point[2] - self.shift:point[2] + self.shift,
                     point[1] - self.shift:point[1] + self.shift,
                     point[0] - self.shift:point[0] + self.shift]

            if self.use_CL_DA:
                img_ice = self.__DA_SelReference(np.array(img_ice))

            img_ice = np.array(img_ice).reshape(-1, self.shift * 2, self.shift * 2, self.shift * 2)
            img_ice = torch.as_tensor(img_ice).float()

        if self.use_bg_part and self.Sel_Referance:
            if self.use_ice_part:
                return img, img_bg, img_ice
            else:
                return img, img_bg, position
        else:
            return img, label, position

    def __len__(self):
        return max(len(self.coords), len(self.data))
        # return min(len(self.coords), len(self.data))

    def __rotation3D(self, data, degree):
        data = np.rot90(data, degree[0], (0, 1))
        data = np.rot90(data, degree[1], (1, 2))
        data = np.rot90(data, degree[2], (0, 2))
        return data

    def __DA_SelReference(self, data):
        D, H, W = data.shape
        out = data.reshape(1, D, H, W)
        out = np.concatenate([out, np.rot90(data, 1, (0, 2)).reshape(1, D, H, W)], axis=0)
        out = np.concatenate([out, np.rot90(data, 2, (0, 2)).reshape(1, D, H, W)], axis=0)
        out = np.concatenate([out, np.rot90(data, 3, (0, 2)).reshape(1, D, H, W)], axis=0)
        out = np.concatenate([out, np.rot90(data, 1, (1, 2)).reshape(1, D, H, W)], axis=0)
        out = np.concatenate([out, np.rot90(data, 3, (1, 2)).reshape(1, D, H, W)], axis=0)
        for axis_idx in range(out.shape[0]):
            data = out[axis_idx]
            out = np.concatenate([out, data[::-1, :, :].reshape(1, D, H, W)], axis=0)
            for idx in range(1, 4):
                out = np.concatenate([out, np.rot90(data, idx, (0, 1)).reshape(1, D, H, W)], axis=0)
                out = np.concatenate([out, np.rot90(data, idx, (0, 1)).reshape(1, D, H, W)[:, ::-1, :, :]], axis=0)
        return out

    def __DA_SelReference_inital(self, data):
        out = data
        for idx in range(1, 4):
            out = np.concatenate([out, np.rot90(data, idx, (0, 1))], axis=0)
            out = np.concatenate([out, np.rot90(data, idx, (1, 2))], axis=0)
            out = np.concatenate([out, np.rot90(data, idx, (0, 2))], axis=0)
        out = np.concatenate([out, data[::-1, :, :]], axis=0)
        out = np.concatenate([out, data[:, ::-1, :]], axis=0)
        out = np.concatenate([out, data[:, :, ::-1]], axis=0)
        return out


    def __sample(self, point, bound):
        # point: z, y, x
        new_point = point + np.random.randint(-self.radius, self.radius + 1, size=3)
        new_point[new_point < self.shift] = self.shift
        new_point[new_point + self.shift > bound] = bound[new_point + self.shift > bound] - self.shift
        return new_point


# transform label to multichannel
def multiclass_label(x, num_classes, first_idx=0):
    for i in range(first_idx, first_idx + num_classes):
        label_temp = x
        label_temp = np.where(label_temp == i, 1, 0)
        if i == first_idx:
            label_new = label_temp
        else:
            label_new = np.concatenate((label_new, label_temp))

    return label_new


if __name__ == '__main__':
    num_cls = 13
    dataloader = DataLoader(Dataset_ClsBased(mode='val',
                                             block_size=32,
                                             num_class=num_cls,
                                             random_num=0,
                                             use_bg=False,
                                             test_use_pad=False, pad_size=18,
                                             data_split=[6, 6, 6],
                                             base_dir="/ldap_shared/synology_shared/shrec_2020/shrec2020_new",
                                             label_name="sphere7",
                                             coord_format=".coords",
                                             tomo_format='.mrc',
                                             norm_type='normalization'),
                            batch_size=1,
                            num_workers=1,
                            shuffle=False,
                            pin_memory=False)
    import matplotlib.pyplot as plt

    rows = 2
    cols = 11
    plt.figure(1, figsize=(cols * 7, rows * 7))
    for idx, (img, label, position) in enumerate(dataloader):
        if idx == 100:
            print(position)
            print(img.shape)
            print(img.max(), img.min())
            print(label.max(), label.min())
            print(label.shape)

            imgs = img[0, 0, 0:31:3, ...]
            labels = label[0, 0, 0:31:3, ...]
            z, h, w = imgs.shape
            for i in range(z):
                plt.subplot(rows, cols, i + 1)
                plt.imshow(imgs[i, ...], cmap=plt.cm.gray)
                plt.axis('off')

                plt.subplot(rows, cols, i + 1 + cols)
                plt.imshow(labels[i, ...], cmap=plt.cm.gray)
                plt.axis('off')

            plt.tight_layout()
            plt.savefig('temp.png')
            print(position)
            break
