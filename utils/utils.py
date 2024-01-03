import numpy as np
import pandas as pd
import cv2
import math
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, QObject, pyqtSignal
import threading, inspect
import ctypes

def rescale(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min)


def hist_equ(tomo, min_val=0.01, max_val=0.99):
    fre_num, xx, _ = plt.hist(tomo.flatten(), bins=256, cumulative=True)
    fre_num /= fre_num[-1]
    for idx, x in enumerate(fre_num):
        if x > min_val:
            min_idx = int(idx)
            break

    for idx, x in enumerate(fre_num[::-1]):
        if x < max_val:
            max_idx = int(len(fre_num) - idx)
            break
    tomo = np.clip(tomo, xx[min_idx], xx[max_idx])
    # tomo = np.clip(tomo, 52, 150)
    # print(xx[min_idx], xx[max_idx])
    return tomo


def gauss_filter(kernel_size=3, sigma=1):
    max_idx = kernel_size // 2
    idx = np.linspace(-max_idx, max_idx, kernel_size)
    Y, X = np.meshgrid(idx, idx)
    gauss_filter = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    gauss_filter /= np.sum(np.sum(gauss_filter))
    return gauss_filter


def stretch(tomo):
    tomo = (tomo - np.min(tomo)) / (np.max(tomo) - np.min(tomo)) * 255
    return np.array(tomo).astype(np.uint8)


class myThread(threading.Thread):
    def __init__(self, thread_id, func, args, emit_str):
        threading.Thread.__init__(self)
        self.threadID = thread_id
        self.func = func
        self.args = args
        self.emit_str = emit_str
        self.n = 1

    def run(self):
        while self.n:
            self.pid_num = self.func(self.args, self.emit_str)
            self.n -= 1

    def get_n(self):
        return self.n


def make_video(tomo, save_path, fps, size):
    if 'mp4' in save_path:
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    elif 'avi' in save_path:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    videowriter = cv2.VideoWriter(save_path,
                                  fourcc,
                                  fps,
                                  size)
    for i in range(tomo.shape[0]):
        img = tomo[i]
        videowriter.write(img)


class Concur(threading.Thread):
    """
    停止thread
    """
    def __init__(self, job, args, stdout):
        super(Concur, self).__init__()
        self.__flag = threading.Event()
        self.__flag.set()
        self.__running = threading.Event()
        self.__running.set()
        self.job = job
        self.args = args
        self.stdout = stdout

    def run(self):
        while self.__running.isSet():
            self.__flag.wait()
            try:
                self.job(self.args, self.stdout)
                self.pause()
            except Exception as e:
                self.stop()
                print(e)

    def pause(self):
        self.__flag.clear()

    def resume(self):
        self.__flag.set()

    def stop(self):
        self.__flag.set()
        self.__running.clear()


def _async_raise(tid, exctype):
    """
    stop thread
    """
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class EmittingStr(QObject):
    textWritten = pyqtSignal(str)

    def __init__(self):
        super(EmittingStr, self).__init__()

    def write(self, text):
        try:
            if len(str(text)) >= 2:
                self.textWritten.emit(str(text))
        except:
            pass

    def flush(self, text=None):
        pass


class ThreadShowInfo(QThread):
    def __init__(self, func, args):
        super(ThreadShowInfo, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(self.args)


def add_transparency(img, label, factor, color, thresh):
    img = np.array(255.0 * rescale(img), dtype=np.uint8)

    alpha_channel = np.ones(img.shape, dtype=img.dtype) * 255

    img_BGRA = cv2.merge((img, img, img, alpha_channel))

    img = img[:, :, np.newaxis]
    img1 = img.repeat([3], axis=2)
    img1[label.astype(int) == 1] = (255, 0, 0)
    img1[label.astype(int) == 2] = (0, 255, 0)
    img1[label.astype(int) == 3] = (0, 0, 255)
    img1[label.astype(int) == 4] = (0, 255, 255)
    # img1[label > factor] = color
    c_b, c_g, c_r = cv2.split(img1)
    mask = np.where(label > factor, 1, 0)
    img1_alpha = np.array(mask * 255 * factor, dtype=np.uint8)
    img1_alpha[img1_alpha == 0] = 255
    img1_BGRA = cv2.merge((c_b, c_g, c_r, img1_alpha))

    out = cv2.addWeighted(img_BGRA, 1 - factor, img1_BGRA, factor, 0)

    return np.array(out)


def annotate_particle(img, coords, diameter, zz, idx, circle_width, color):
    """Plot circle centered at the particle coordinates directly on the tomogram."""
    img = 255.0 * rescale(img)
    img = img[:, :, np.newaxis]
    rgb_uint8 = img.repeat([3], axis=2).astype(np.uint8)

    if idx == 0:
        columns = ['x', 'y', 'z']
    elif idx == 1:
        columns = ['z', 'y', 'x']
    else:
        columns = ['x', 'z', 'y']

    df = pd.DataFrame(data=coords,
                      columns=columns)
    color = np.array(color)
    print(color.shape)
    df["R"] = color[:, 0]
    df["G"] = color[:, 1]
    df["B"] = color[:, 2]

    r = diameter / 2
    coords_xy = df[df['z'] >= (zz - r)]
    coords_xy = coords_xy[df['z'] <= (zz + r)]
    rr2 = r ** 2 - (zz - coords_xy['z']) ** 2
    rr = [math.sqrt(i) for i in rr2]
    for x, y, rad, r, g, b in zip(coords_xy['x'], coords_xy['y'], rr, coords_xy['R'], coords_xy['G'], coords_xy['B']):
        cv2.circle(rgb_uint8, (int(y), int(x)), int(rad), (r, g, b), circle_width)
    return rgb_uint8