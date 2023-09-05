# The MIT License (MIT)
#
# Copyright (c) 2017 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

import sys

from matplotlib import pyplot as plt
from collections import deque

from winsound import Beep
from time import sleep

import myo
import numpy as np
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import *


names = ['rest', 'fist', 'spread', 'in', 'out']


class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.data = deque(maxlen=n)
        self.clf = None
        self.s = None
        self.las = -1
        self.cnt = [0] * 10

    def get_emg_data(self):
        return list(self.data)

    def on_arm_synced(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        self.data.append(event.emg)
        if self.clf is None:
            return
        
        if len(self.data) < 10:
            return
        ls = list(self.data)
        raw = np.array(ls[-10:]).transpose()
        ft = getff(raw)
        global names
        id = self.clf.predict([ft])[0]

        if id != self.las:
            self.cnt[id] = 0
        else:
            self.cnt[id] += 1
        self.las = id
        print(names[id], self.cnt[id])
        if self.s is not None:
            if id == 0 and self.cnt[id] == 400:
                sendmsg(self.s, id + 1)
            if 1 <= id <= 2 and self.cnt[id] == 50:
                sendmsg(self.s, id + 1)
            if id >= 3 and self.cnt[id] >= 10 and self.cnt[id] % 10 == 0:
                sendmsg(self.s, id + 1)


def getdata(cl, dc, id):
    global names
    print(f'record start {names[id]}')
    Beep(400, 250)
    sleep(4)
    data = np.array(cl.get_emg_data())
    Beep(800, 250)
    print('record end')
    dc[id] = data


if len(sys.argv) < 2:
    print('require <mode>')
    quit()
assert sys.argv[1] in ('train', 'test')
mode = sys.argv[1]

myo.init(sdk_path="D:/others/myo-sdk-win-0.9.0")
hub = myo.Hub()
listener = EmgCollector(600)

remote = False
if len(sys.argv) > 2 and sys.argv[2] == 'remote':
    remote = True

# ip = '192.168.43.143'   # wlan
ip = '169.254.18.197'   # eth
port = 29242

with hub.run_in_background(listener.on_event):
    print("starting")
    while len(listener.data) == 0:
        pass
    print("synced!")

    if mode == 'train':
        dc = {}
        for i in range(len(names)):
            getdata(listener, dc, i)
        clf = train(dc)
        with open('lda.dat', 'wb') as f:
            pickle.dump(clf, f)
    
    if mode == 'test':
        if remote:
            s = getsock(ip, port)
            listener.s = s
        with open('lda.dat', 'rb') as f:
            listener.clf = pickle.load(f)
        input()
        sendmsg(listener.s, 0)
        s.close()
