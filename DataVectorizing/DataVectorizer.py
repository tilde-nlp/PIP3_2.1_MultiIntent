import sys
import os
import numpy as np
from itertools import chain
import argparse
import datetime
import subprocess
import re
import DataVectorizing.VectorizerConnector
from DataVectorizing.VectorizerConnector import VectorizerConnector


class Vectorizer:
    def __init__(self, address=None, port=None, lru_cache_size=10000):
        if address is not None:
            self.vc = VectorizerConnector(address, port, lru_cache_size)

    # sentence -> vector of len maxWords (30)
    def getX(self, question, maxWords=30):
        wordVectors = self.vc.vectorize(question)
        if len(wordVectors) == 0:
            # workaround for the case when the tokenized question has no words
            testcase = self.vc.vectorize("a")
            dim = len(testcase[0])
        else:
            dim = len(wordVectors[0])
    
        def getZeroVector(dim):
            return [0] * dim
        def appendZeroVector(x, N, dim):
            if len(x) > N:
                return x[:N]
            return x + [getZeroVector(dim)] * (N - len(x))

        wordVectors = appendZeroVector(wordVectors, maxWords, dim)
        X = np.array(wordVectors)
        return X

    def set_intents(self, intents):
        self.answerDict = { ans : i for i, ans in enumerate(sorted(set(intents))) }

    def getY(self, intent):
        retval = np.zeros(len(self.answerDict))
        if type(intent) is list:
            for item in intent:
                if item in self.answerDict.keys():
                    retval[self.answerDict[item]] = 1
        else:
            if intent in self.answerDict.keys():
                retval[self.answerDict[intent]] = 1
        return retval

    def getY_all(self, intents):
        answerInts = [ answerDict[ans] for ans in answers ]
        retval = np.eye(len(self.answerDict))[answerInts]
        return retval




