import sys
import os
import numpy as np
import json
import keras
import datetime
import random
import math

import tensorflow as tf

from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import (Dense, Activation, Conv1D, MaxPooling1D, 
                          Dropout, Flatten, Input, Add, Average,
                          LSTM, Lambda, AveragePooling1D,
                          Reshape)
from tensorflow.keras.optimizers import SGD, Adam
from keras import regularizers
from keras import backend as K
from sklearn.model_selection import StratifiedKFold


import argparse

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")

def log(s):
    print(ts() + s, flush=True)

def createAdamOptimizer(lr=0.001, beta_1=0.9, beta_2=0.999,
                        decay=0, epsilon=None, amsgrad=False):
    opt = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2,
               epsilon=epsilon, decay=decay, amsgrad=amsgrad, )
    return opt

# This model is not used (as CNN is better).
# This would require sentence vector as input (instead of word vectors)
def createModel1(vectorSpaceSize, nClasses, **kwargs):
    model = Sequential()
    model.add(Dense(nClasses, input_dim=vectorSpaceSize, activation='softmax'))
    return model

def createModelCNN(vectorSpaceSize, nClasses, filterCounts=[300, 300], regParam=0, regParam2=0, dropout=0.5, **kwargs):
    inp = Input(shape=(30, vectorSpaceSize))

    filterLayer = []
    for ws, filters in enumerate(filterCounts, start=1):
        if filters > 0:
            conv = Conv1D(filters=filters,
                         kernel_size=ws,
                         activation='relu',
                         kernel_regularizer=regularizers.l1(regParam)
                         )(inp)
            conv = MaxPooling1D(pool_size=30 - ws + 1)(conv)
            filterLayer.append(conv)

    if len(filterLayer) > 1:
        merged = keras.layers.concatenate(filterLayer)
    else:
        merged = filterLayer[0]
    merged = Flatten()(merged)
    
    #out = Dense(units=nClasses, activation='sigmoid')(merged)
    if dropout>0:
        merged = Dropout(rate=dropout)(merged)
    out = Dense(units=nClasses, activation='softmax', kernel_regularizer=regularizers.l2(regParam2))(merged)

    model = Model(inp, out)
       
    return model

def createModelCNNMultiOneV1(vectorSpaceSize, nClasses, filterCounts=[300, 300], regParam=0, regParam2=0, dropout=0.5, **kwargs):
    inp = Input(shape=(30, vectorSpaceSize))

    filterLayer = []
    for ws, filters in enumerate(filterCounts, start=1):
        if filters > 0:
            conv = Conv1D(filters=filters,
                         kernel_size=ws,
                         activation='relu',
                         kernel_regularizer=regularizers.l1(regParam)
                         )(inp)
            conv = MaxPooling1D(pool_size=30 - ws + 1)(conv)
            filterLayer.append(conv)

    if len(filterLayer) > 1:
        merged = keras.layers.concatenate(filterLayer)
    else:
        merged = filterLayer[0]
    merged = Flatten()(merged)
    
    if dropout > 0:
        merged = Dropout(rate=dropout)(merged)
    outLayer = []
    for i in range(nClasses):
        out = Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(regParam2))(merged)
        outLayer.append(out)
    
    out = keras.layers.concatenate(outLayer)
    out = Lambda(lambda x: x[:,::2])(out)
    model = Model(inp, out)
       
    return model


def createModelCNNMultiOneV2(vectorSpaceSize, nClasses, filterCounts=[300, 300], regParam=0, regParam2=0, dropout=0.5, **kwargs):
    inp = Input(shape=(30, vectorSpaceSize))

    filterLayer = []
    for ws, filters in enumerate(filterCounts, start=1):
        if filters > 0:
            conv = Conv1D(filters=filters,
                         kernel_size=ws,
                         activation='relu',
                         kernel_regularizer=regularizers.l1(regParam)
                         )(inp)
            conv = MaxPooling1D(pool_size=30 - ws + 1)(conv)
            filterLayer.append(conv)

    if len(filterLayer) > 1:
        merged = keras.layers.concatenate(filterLayer)
    else:
        merged = filterLayer[0]
    merged = Flatten()(merged)
    
    if dropout > 0:
        merged = Dropout(rate=dropout)(merged)
    outLayer = []
    out = Dense(units=nClasses, activation='sigmoid', kernel_regularizer=regularizers.l2(regParam2))(merged)

    model = Model(inp, out)
       
    return model

def readDict(fileName):
    dd = dict()
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.rstrip("\n").split("\t")
            dd[s[0]] = int(s[1])
    return dd

def reverseDict(answerDict):
    revDict = {value : key for key, value in answerDict.items()}
    return revDict
    
 
def settleConflict(resultArray):
    res = np.argmax(resultArray)
    return res

def releaseMemory():
    K.clear_session()
    
def performMultiTest(trainX, trainY, testX, testY, intentmap, label="",
               simpleNN=False, epochs=None, dontUseTimestamp=False,
               multi=True, modelVersion=0, **kwargs):

    releaseMemory()

    revMap = reverseDict(intentmap)

    assert("__other__" not in intentmap.keys())
    models = {}
    results = {}
    
    startTime = datetime.datetime.now()
    dateText = startTime.strftime("%Y-%m-%d")
    timeText = startTime.strftime("%H-%M-%S")
    if dontUseTimestamp:
        baseLogDir = ("./tbGraph/"
                    + label
                    )
    else:
        baseLogDir = ("./tbGraph/"
                    + dateText + "/" + label)
                  #  + timeText 
                   # + "_" + label
                   # )
    
    if multi:
    
        for i, intent in revMap.items():
            currentIntentTrainY = np.eye(2)[(trainY.argmax(axis=1)==i).astype(int)]
            currentIntentTestY = np.eye(2)[(testY.argmax(axis=1)==i).astype(int)]
            currentIntentIntentmap = { "__other__": 0, intent: 1}
        
            logDir = baseLogDir + f"/intents/int{i:03d}"
            releaseMemory()

            model, res = performTest(trainX, currentIntentTrainY, testX, currentIntentTestY, currentIntentIntentmap, logDir=logDir,
                   simpleNN=simpleNN, epochs=epochs, **kwargs)
            models[i] = model
            results[i] = res
    else:
        logDir = baseLogDir + f"/all_intents"
        model, res = performTest(trainX, trainY, testX, testY, intentmap, logDir=logDir,
                   simpleNN=simpleNN, epochs=epochs, modelVersion=modelVersion, **kwargs)
        models = model
        results = res
        pass
    
    endTime = datetime.datetime.now()
    
    if multi:
        rr = np.zeros((len(results[0]), len(results)))
        for i, r in results.items():
            for exnr, ex in enumerate(r):
                rr[exnr][i] = ex[1] # ex[1] is the given probability for the class (ex[0] is "__other__")
    else:
        rr = results
        pass
    testAccuracy = np.sum(rr.argmax(axis=1)==testY.argmax(axis=1))/len(testY)
    
    resultJson = {
        "test_accuracy": testAccuracy
    }
    with open(os.path.join(baseLogDir, "results.txt"), "w", encoding="utf-8") as f:
        json.dump(resultJson, f, indent=2)
    
    
    with open(os.path.join(baseLogDir, "confidenceSumFile.txt"), "w", encoding="utf-8") as f:
        print("sum", "max", "rest", sep="\t", file=f)
        sums = rr.sum(axis=1).tolist()
        maxs = rr.max(axis=1).tolist()
        rest = (np.array(sums) - np.array(maxs)).tolist()
        for s, m, r in zip(sums, maxs, rest):
            print(s, m, r, sep="\t", file=f)

    if not dontUseTimestamp:        
        with open('./tbGraph/multi_results.txt', 'a', encoding='utf-8', buffering=1) as resFile:
            resFile.write(dateText + " " + timeText + "\t")
            resFile.write(label + "\t")
            resFile.write(str(endTime - startTime) + "\t")
        
            resFile.write(f"{testAccuracy}")
      
            #for key in ['acc', 'val_acc']:
            #    if (key in history.history.keys()):
            #        val = np.average(history.history[key][-5:])
            #    else:
            #        val = "-"
            #    resFile.write(str(val) + "\t")
            #resFile.write(', '.join(
            #    ["{} = {}".format(key, value) 
            #        for (key, value) in sorted(params.items())]))
        
        
            resFile.write("\n")

        
    return models, results
    
class CustomCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        print(". ")
    def on_epoch_end(self, epoch, logs=None):
        if epoch in [19, 39, 59, 79, 99]:
            with open('./tbGraph/xval_multi_partial_results.txt', 'a', encoding='utf-8', buffering=1) as resFile:
                resFile.write("\t")
                resFile.write(f"vvv ep{epoch+1}\t")
                resFile.write("\t")
                resFile.write(f"{logs.get('val_acc')}")
                resFile.write("\n")

        print(f"{epoch%10}", end="")
        loss = logs.get("loss")
        if math.isnan(loss):
            print(f"Loss is NaN, stopping")
            self.model.stop_training = True


def performTest(trainX, trainY, testX, testY, intentmap, logDir="./tbG/",
               simpleNN=False, epochs=None, modelVersion=0, **kwargs):

    revMap = reverseDict(intentmap)

    if epochs is None:
        epochs = 200 if simpleNN else 40 # more epochs

    vectorSpaceSize = trainX.shape[-1]
    nClasses = trainY.shape[1]

    
    os.makedirs(logDir, exist_ok=True)
    with open(os.path.join(logDir, 'params.txt'), 'w', encoding='utf-8', buffering=1) as out:
        #log("start creating model")
    
        if simpleNN:
            model = createModel1(vectorSpaceSize, nClasses)
            optimizer = createAdamOptimizer(lr = 0.03)
        else:
            if modelVersion == 0:
                model = createModelCNN(vectorSpaceSize, nClasses)
            elif modelVersion == 1:
                model = createModelCNNMultiOneV1(vectorSpaceSize, nClasses, **kwargs)
            elif modelVersion == 2:
                model = createModelCNNMultiOneV2(vectorSpaceSize, nClasses, **kwargs)
            optimizer = createAdamOptimizer()
        
        model.compile(optimizer=optimizer,
                      #loss='categorical_crossentropy',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        plot_model(model, to_file=os.path.join(logDir, 'model.png'))

        #log("start training")

        tbCallBack = keras.callbacks.TensorBoard(log_dir=logDir, histogram_freq=0, # histogram_freq=1,
                write_graph=True, write_images=True)

    
        out.write(model.to_json() + "\n\n")
        model.summary(print_fn=lambda x: out.write(x + '\n'))
        out.write("\n")
    
    history = model.fit(trainX, trainY, epochs=epochs, verbose=0,
        callbacks=[tbCallBack, CustomCallback()],
        validation_data=(testX, testY),
    )
    model.save(os.path.join(logDir, "model.hdf"))

    res = model.predict(testX)
    with open(os.path.join(logDir, "raw_preds.txt"), "w", encoding="utf-8") as f:
        for i in range(len(res)):
            json.dump(res[i].tolist(), f)
            f.write("\n")


    winnerIntents = np.argmax(res, 1)
    winnerIntentScores = np.max(res, 1) #would be nicer to get both top intents and scores in one go
    with open(os.path.join(logDir, "predictions.txt"), "w", encoding="utf-8") as f:
        for wi, wis in zip(winnerIntents, winnerIntentScores):
            print(revMap[wi], wis, sep="\t", file=f)

    for key in history.history.keys():
        with open(logDir + '/hist_' + key + '.txt', 'w', encoding='utf-8', buffering=1) as histFile:
            for val in history.history[key]:
                histFile.write(str(val) + "\n")

    return model, res


def loadStuff():
    trainX = np.load(trainQuestionFile)
    trainY = np.load(trainAnswerFile)
    testX = np.load(testQuestionFile)
    testY = np.load(testAnswerFile)
    intentmap = readDict(intentmapFile)
    revMap = reverseDict(intentmap)

def getTestExamples(unshuffledX, unshuffledY, intentmap, Xtexts, label=None, logDir=None, k=5, **kwargs):
    # have to have Xtexts
    X, y, kf = getKFold(unshuffledX, unshuffledY, k=k)
    for foldNr, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    pass


def performXValMultiTest(unshuffledX, unshuffledY, intentmap, label, k=5, multi=True, modelVersion=0, **kwargs):
    log(label)
    minCount = np.unique(unshuffledY.argmax(axis=1), return_counts=True)[1].min()
    if minCount < k:
        log(f"At least {k} test examples for each intent are needed to perform cross-validation (currently minimum is {minCount})")
        return np.zeros(0, dtype=int)
    X, y, kf = getKFold(unshuffledX, unshuffledY, k=k)
    confMatrix = np.zeros((len(y[0]), len(y[0])), dtype=int)

    all_rr = np.zeros((0,len(y[0])))
    all_rr_true = np.zeros((0,len(y[0])))

    startTime = datetime.datetime.now()
    dateText = startTime.strftime("%Y-%m-%d")
    timeText = startTime.strftime("%H-%M-%S")


    for foldNr, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        new_label = (dateText + "/" + label + f"/fold{foldNr}"
                    )

        if multi:
            models, results = performMultiTest(X_train, y_train, X_test, y_test, intentmap, label=new_label, dontUseTimestamp=True, **kwargs)

            rr = np.zeros((len(results[0]), len(results)))
            for i, r in results.items():
                for exnr, ex in enumerate(r):
                    rr[exnr][i] = ex[1] # ex[1] is the given probability for the class (ex[0] is "__other__")

        else:
            models, results = performMultiTest(X_train, y_train, X_test, y_test,
                                               intentmap, label=new_label,
                                               dontUseTimestamp=True, multi=False, modelVersion=modelVersion,
                                               **kwargs)

            rr = results # probably this should suffice

        all_rr = np.vstack((all_rr, rr))
        all_rr_true = np.vstack((all_rr_true, y_test))

    testAccuracy = np.sum(all_rr.argmax(axis=1)==all_rr_true.argmax(axis=1))/len(all_rr_true) #!!!aprēķina, vai visvarbūtiskākais intents pareizs
    
    topIntent = np.argmax(all_rr, axis=1)
    correctIntent = np.argmax(all_rr_true, axis=1)
    for j in range(0, len(topIntent)):
        confMatrix[correctIntent[j], topIntent[j]] += 1
    

    endTime = datetime.datetime.now()
    baseLogDir = ("./tbGraph/"
            + dateText + "/" + label
            )

    
    resultJson = {
        "test_accuracy": testAccuracy
    }
    with open(os.path.join(baseLogDir, "results.txt"), "w", encoding="utf-8") as f:
        json.dump(resultJson, f, indent=2)
    
    
    with open(os.path.join(baseLogDir, "confidenceSumFile.txt"), "w", encoding="utf-8") as f:
        print("sum", "max", "rest", sep="\t", file=f)
        sums = all_rr.sum(axis=1).tolist()
        maxs = all_rr.max(axis=1).tolist()
        rest = (np.array(sums) - np.array(maxs)).tolist()
        for s, m, r in zip(sums, maxs, rest):
            print(s, m, r, sep="\t", file=f)

    with open('./tbGraph/xval_multi_partial_results.txt', 'a', encoding='utf-8', buffering=1) as resFile: # for other epoch counts
        resFile.write(dateText + " " + timeText + "\t")
        resFile.write(label + "\t")
        resFile.write(str(endTime - startTime).split('.')[0] + "\t")
        resFile.write(f"{testAccuracy}")
        resFile.write("\n")
        
    with open('./tbGraph/xval_multi_results.txt', 'a', encoding='utf-8', buffering=1) as resFile:
        resFile.write(dateText + " " + timeText + "\t")
        resFile.write(label + "\t")
        resFile.write(str(endTime - startTime).split('.')[0] + "\t")
        resFile.write(f"{testAccuracy}")
        resFile.write("\n")

    with open(os.path.join(baseLogDir, "confusion.txt"), "w", encoding="utf-8") as confMatrFile:
        confMatrFile.write("\n".join("\t".join(str(x) for x in y) for y in confMatrix) + "\n")

    return


def getXValTestSets(unshuffledX, unshuffledY, unshuffledExamples, intentmap, k=5):
    revMap = reverseDict(intentmap)

    X, y, ex, kf = getKFold(unshuffledX, unshuffledY, unshuffledExamples, k=k)

    #outX = []
    outY = []
    outEx = []
    outAns = []
    
    for foldNr, (train_index, test_index) in enumerate(kf):
        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]

        for ti in test_index:
            outEx.append(ex[ti])
            outAns.append(revMap[np.argmax(y[ti])])
            outY.append(y[ti])


    return outEx, outAns, outY

def getKFold(X, y, examples=None, k=10, randomSeed=1):
    assert(len(X) == len(y))
    n = len(y)
    r = list(range(n))
    random.seed(randomSeed)
    random.shuffle(r)

    #own shuffle
    shX = []; shY = [];
    if examples is not None:
        ex = []
    for pos in r:
        shX.append(X[pos])
        shY.append(y[pos])
        if examples is not None:
            ex.append(examples[pos])

        
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=randomSeed)
    shX, shY = np.array(shX), np.array(shY)
    n_samples = len(shX)
    res = skf.split(np.zeros(n_samples), shY.argmax(axis=1))
    if examples is not None:
        return shX, shY, ex, res
    else:
        return shX, shY, res


def main():

    pass

if __name__ == "__main__":
    sys.exit(int(main() or 0))