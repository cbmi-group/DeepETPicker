from multiprocessing import Pool
import mrcfile
import numpy as np
import warnings
import os
import glob
import sys

warnings.simplefilter('ignore')
class InputNorm():
    def __init__(self, tomo_path, tomo_format, base_dir, norm_type):
        self.tomo_path = tomo_path
        self.tomo_format = tomo_format
        self.base_dir = base_dir
        self.norm_type = norm_type


        if self.norm_type == 'standardization':
            self.save_dir = os.path.join(self.base_dir, 'data_std')
        elif self.norm_type == 'normalization':
            self.save_dir = os.path.join(self.base_dir, 'data_norm')
        os.makedirs(self.save_dir, exist_ok=True)

        self.dir_list = [i.split('/')[-1] for i in glob.glob(self.tomo_path + '/*%s' % self.tomo_format)]
        print(self.dir_list)

    def single_handle(self, i):
        dir_name = self.dir_list[i]
        with mrcfile.open(os.path.join(self.tomo_path, dir_name),
                          permissive=True) as gm:
            try:
                data = np.array(gm.data).astype(np.float)
            except:
                data = np.array(gm.data).astype(np.float32)
            # print(data.shape)
            if self.norm_type == 'standardization':
                data -= data.mean()
                data /= data.std()
            elif self.norm_type == 'normalization':
                data -= data.min()
                data /= (data.max() - data.min())

            reconstruction_norm = mrcfile.new(
                os.path.join(self.save_dir, dir_name), overwrite=True)
            try:
                reconstruction_norm.set_data(data.astype(np.float32))
            except:
                reconstruction_norm.set_data(data.astype(np.float))

            reconstruction_norm.close()
            print('%d/%d finished.' % (i + 1, len(self.dir_list)))

    def handle_parallel(self):
        with Pool(len(self.dir_list)) as p:
            p.map(self.single_handle, np.arange(len(self.dir_list)).tolist())


def norm_show(args):
    tomo_path, tomo_format, base_dir, norm_type, stdout = args
    if stdout is not None:
        save_stdout = sys.stdout
        save_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout

    pre_norm = InputNorm(tomo_path, tomo_format, base_dir, norm_type)
    pre_norm.handle_parallel()
    print('Standardization finished!')
    print('*' * 100)
    """
    try:
        pre_norm = InputNorm(tomo_path, tomo_format, base_dir, norm_type)
        pre_norm.handle_parallel()
        print('Standardization finished!')
        print('*' * 100)
    except:
        stdout.flush()
        stdout.write('Normalization Exception!')
        return 0
    """
    if stdout is not None:
        sys.stderr = save_stderr
        sys.stdout = save_stdout