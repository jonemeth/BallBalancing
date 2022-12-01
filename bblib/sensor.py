import time
import threading
from typing import Tuple

import serial
import numpy as np


def lin_reg(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(np.square(x))
    sxy = np.sum(np.multiply(x, y))
    N = len(x)

    denom = (N * sx2 - sx ** 2)

    if abs(denom) < 0.0001:
        print('para', flush=True)
        m = 0.0
    else:
        m = (N * sxy - sx * sy) / denom
    b = (sy - m * sx) / N

    return m, b


ser = serial.Serial('/dev/ttyACM0', 9600)


class Sensor:
    def __init__(self, multipliers: Tuple[float, float]):
        self.multipliers = multipliers
        self.thread = threading.Thread(target=self.worker)
        self.ser = ser
        self.ser.reset_input_buffer()
        self.thread.start()

        self.listX = []
        self.listY = []
        self.listT = []
        self.lock = threading.Lock()

    def worker(self):

        while True:
            try:
                self.ser.reset_input_buffer()
                self.ser.readline()
                r = self.ser.readline().decode("utf-8")
                result = r.split(" , ")
            except:
                continue

            if 2 != len(result):
                continue

            try:
                x = int(result[0]) / 1000 * self.multipliers[0]
                y = int(result[1]) / 1000 * self.multipliers[1]
            except:
                continue

            self.lock.acquire()

            try:
                t = time.time()
                while len(self.listT) > 3 and t - self.listT[0] > 0.2:
                    self.listT.pop(0)
                    self.listX.pop(0)
                    self.listY.pop(0)

                self.listT.append(t)
                self.listX.append(x)
                self.listY.append(y)

            finally:
                self.lock.release()

    def get(self):
        if len(self.listT) <= 0:
            return 0, 0, 0, 0, 0, 0

        self.lock.acquire()

        try:
            T = time.time()

            rel_list_t = (np.array(self.listT) - self.listT[0]).tolist()
            mx, bx = lin_reg(rel_list_t, self.listX)
            my, by = lin_reg(rel_list_t, self.listY)

            # filter
            if len(self.listT) > 3:

                diffs_x = [mx * rel_list_t[i] + bx - self.listX[i] for i in range(len(rel_list_t))]
                diffs_y = [my * rel_list_t[i] + by - self.listY[i] for i in range(len(rel_list_t))]
                dx_std = np.std(diffs_x)
                dy_std = np.std(diffs_y)

                f_tx = [rel_list_t[i] for i in range(len(rel_list_t)) if abs(diffs_x[i]) < 2.0 * dx_std]
                fX = [self.listX[i] for i in range(len(rel_list_t)) if abs(diffs_x[i]) < 2.0 * dx_std]

                f_ty = [rel_list_t[i] for i in range(len(rel_list_t)) if abs(diffs_y[i]) < 2.0 * dy_std]
                f_y = [self.listY[i] for i in range(len(rel_list_t)) if abs(diffs_y[i]) < 2.0 * dy_std]

                if len(fX) >= 3:
                    mx, bx = lin_reg(f_tx, fX)
                if len(f_y) >= 3:
                    my, by = lin_reg(f_ty, f_y)

                rel_t = T - self.listT[0]

                ball_x = mx * rel_t + bx
                ball_y = my * rel_t + by
                speed_x = mx
                speed_y = my
            else:
                return 0, 0, 0, 0, 0, 0
        finally:
            self.lock.release()

        return self.listX[-1], self.listY[-1], ball_x, ball_y, speed_x, speed_y
