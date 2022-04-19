import sys
import os
from copy import deepcopy
import re


def toLowercaseCharsTokenized(line):
    # \W = non alphanumeric chars
    line = re.sub(r'[\W_]+', ' ', line, flags=re.UNICODE)
    line = re.sub(r' +', ' ', line, flags=re.UNICODE)
    line = line.strip().lower()
    return line


def removeDuplicates(data):
    #TODO:
    pass


def data_preprocess(data):
    retval = []
    for d in data:
        d['text'] = toLowercaseCharsTokenized(d['text'])
        retval.append(d)
    return retval

def string_preprocess(s):
    return toLowercaseCharsTokenized(s)

