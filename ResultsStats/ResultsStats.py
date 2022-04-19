import sys
import os
import numpy as np
import json
import datetime
import random
import math

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import precision_recall_fscore_support

import argparse


def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(s):
    print(f"{ts()}: {s}", flush=True)



def leaveOnlyTop(arr):
    top_idx = np.argmax(raw_data, axis=1)
    top_values = raw_data[np.arange(raw_data.shape[0]), top_idx]
    arr = np.zeros_like(raw_data)
    arr[np.arange(raw_data.shape[0]), top_idx] = top_values

def leaveOnlyThr(arr, t):
    return np.where(arr >= thr, arr, 0)

def binarize(arr, t=None):
    if t is None:
        return np.where(arr > 0, 1, 0)
    else:
        return np.where(arr >= t, 1, 0)


def oh2vec(arr):
    top_idx = np.argmax(arr, axis=1)
    return top_idx
def vec2oh(vec, num_classes):
    arr = np.zeros((vec.size, num_classes))
    arr[np.arange(vec.size), vec] = 1
    return arr



def getStatsSklearn(gt, brp):
    prec, rec, f1, support = precision_recall_fscore_support(gt, brp, average="samples")
    return prec, rec, f1

def getStats(gt, brp):
    #with np.errstate(divide='ignore'):
    numerator = np.sum((gt.astype(bool) & brp.astype(bool)).astype(int), axis=1)
    denum_acc = np.sum((gt.astype(bool) | brp.astype(bool)).astype(int), axis=1)
    denum_prec = np.sum(brp, axis=1)
    denum_rec = np.sum(gt, axis=1)
    prec = np.mean(np.where(denum_prec > 0, numerator / denum_prec, np.where(numerator > 0, 0, 1)))
    rec = np.mean(np.where(denum_rec > 0, numerator / denum_rec, np.where(numerator > 0, 0, 1)))
    acc = np.mean(np.where(denum_acc > 0, numerator / denum_acc, np.where(numerator > 0, 0, 1)))
    f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1, acc 


def getGraph(gt, rp, useOnlyTop=False, steps=101):
    if useOnlyTop:
        rp = leaveOnlyTop(rp)
    arr = []
    for t in np.linspace(0, 1.0, steps):
        brp = binarize(rp, t)
        prec, rec, f1, acc  = getStats(gt, brp)
        value = {"t": t, "prec": prec, "rec": rec, "f1": f1, "acc": acc}
        arr.append(value)
    return arr

def get_pr_curve(raw_data, ground_truth, use_only_top_answer=False):
    if use_only_top_answer:
        top_idx = np.argmax(raw_data, axis=1)
        top_values = raw_data[np.arange(raw_data.shape[0]), top_idx]
        arr = np.zeros_like(raw_data)
        arr[np.arange(raw_data.shape[0]), top_idx] = top_values
    else:
        arr = raw_data
    return arr

    for i in range(len(raw_data)):
        if None:
            pass
        pass
    pass


def main():
    










    pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))