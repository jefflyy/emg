import pickle
import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import socket


def filter(raw):
    sos = signal.butter(5, [20, 99.9999999999999], btype='bandpass', output='sos', fs=200)
    return signal.sosfilt(sos, raw)

def getff(raw):
    # size = (8, 10)
    y = filter(np.abs(raw))
    rms = np.mean(y ** 2, axis=1) ** 0.5
    var = np.var(y, axis=1, ddof=1)
    iav = np.sum(np.abs(y), axis=1)
    ssi = np.sum(y ** 2, axis=1)
    wl = np.sum(np.abs(np.diff(y)), axis=1)
    return np.array([rms, var, iav, ssi, wl]).flatten()

def getface(data, name, x, y):
    n = len(data)
    for i in range(n // 5 - 1):
        st = i * 5
        en = i * 5 + 10
        raw = data[st:en].transpose()
        x.append(getff(raw))
        y.append(name)

def train(dc):
    x = []
    y = []
    for s in dc:
        getface(dc[s], s, x, y)
    clf = LinearDiscriminantAnalysis()
    clf.fit(x, y)
    return clf

def getsock(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s

def sendmsg(s, a):
    s.send(str(a).encode())
