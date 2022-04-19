import sys
import os
import json
from collections import defaultdict
import DataPreprocessing.DataPreprocessing as pre
import DataVectorizing.DataVectorizer as vec
import Classifier.Classifier as cl
import matplotlib.pyplot as plt
import numpy as np
import datetime
from ResultsStats import ResultsStats as rs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv
import pandas as pd
import configparser

def PlotandSave(expectedy, predictedy, filenamelabel, steps=1001):
    arr = rs.getGraph(expectedy, predictedy, steps=steps)
    df = pd.DataFrame(arr)
    df.set_index('t', inplace=True)
    df.to_csv(filenamelabel + "graph.csv", sep="\t")

    prec = [el['prec'] for el in arr]
    rec = [el['rec'] for el in arr]
    f1 = [el['f1'] for el in arr]
    acc = [el['acc'] for el in arr]

    plt.figure()
    plt.plot(prec, label="precision")
    plt.plot(rec, label="recall")
    plt.plot(f1, label="f1")
    plt.plot(acc, label="acc")
    plt.xticks([0,100, 200,300,400,500,600,700,800,900,1000],[0.0,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.vlines([100,500],0,1,linestyles='dotted')
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig(filenamelabel + "prfa.png")

def main():
    try:
        ininame = os.path.dirname(os.path.abspath(__file__))+'\\config.ini'
        config = configparser.ConfigParser()
        config.read(ininame)
        datadir = config['Arguments']['datadir']
        resdir = config['Arguments']['resdir']
        dsName = config['Arguments']['dsName']
        input_file_train = config['Arguments']['input_file_train']
        input_file_test = config['Arguments']['input_file_test']
        lang = config['Arguments']['lang']
        xval = int(config['Arguments']['xval'])
        multi = bool(config['Arguments']['multi'])
        verNr = int(config['Arguments']['verNr'])
        allEp = list(map(int, config['Arguments']['epochs'].split(' ')))

        #Vectorizer
        v = vec.Vectorizer("localhost", 12345, 10**5)

        #Getting train examples 
        intentdict = defaultdict(list)
        with open(input_file_train, "r", encoding="utf-8") as f:
            for line in f:
                cols=line.split('\t')
                if len(cols) == 2:
                    intname = cols[1].rstrip()
                    example = pre.string_preprocess(cols[0].rstrip())
                    if example != "" and example not in intentdict[intname]:
                        intentdict[intname].append(example)

        example2intents = defaultdict(list)
        for intent, examples in intentdict.items():
            if len(examples) >= 5:  #filter out <5 intent data
                for ex in examples:
                    example2intents[ex.strip()].append(intent.strip())

        #The list of intents in training data
        intents = sorted(list(set(ex for item in example2intents for ex in example2intents[item] if len(ex)>0)))
        v.set_intents(intents)

        train_ex = [{"text": key, "intent": example2intents[key], "X": v.getX(key), "Y": v.getY(example2intents[key])} for key in example2intents]

        intentmap = { ans : i for i, ans in enumerate(intents) }

        #Getting test examples  if not x-validation, and training
        if (xval == 0):
            if len(input_file_test) > 0:
                #if there is test file
                example2intents_test = {}

                with open(input_file_test, "r", encoding="utf-8") as f:
                    for line in f:
                        cols=line.split('\t')
                        if len(cols) == 2:
                            example = pre.string_preprocess(cols[0].rstrip())
                            intname = cols[1].rstrip()
                            if example != "" and example not in example2intents_test:
                                example2intents_test[example] = []
                            if intname in intents:
                                example2intents_test[example].append(intname)

                test_ex = [{"text": key, "intent": example2intents_test[key], "X": v.getX(key), "Y": v.getY(example2intents_test[key])} for key in example2intents_test]
                
                trainX = np.array([ex['X'] for ex in train_ex])
                trainY = np.array([ex['Y'] for ex in train_ex])
                testX = np.array([ex['X'] for ex in test_ex])
                testY = np.array([ex['Y'] for ex in test_ex])
                
            else:
                #take 80% for train and 20% for test
                allX = np.array([ex['X'] for ex in train_ex])
                allY = np.array([ex['Y'] for ex in train_ex])
                trainX, testX, trainY, testY = train_test_split(allX, allY, test_size=0.20, random_state=1, stratify=allY)
               
            label = f"{dsName}_orig{verNr}"
            print(label, end=" --- ")
            print(f"train: {len(trainX)}, test: {len(testX)}")
            models, results = cl.performMultiTest(trainX, trainY, testX, testY, intentmap, label,multi=multi, modelVersion=verNr)

            if multi:
                rr = np.zeros((len(results[0]), len(results)))
                for i, r in results.items():
                    for exnr, ex in enumerate(r):
                        rr[exnr][i] = ex[1]
            else:
                rr = results

            # Plotting confidence summs that represent the number of detected intents per test example
            summaryconfidence = np.flip(np.sort([e for e in rr.sum(axis=1).tolist()]))
            confidence = np.flip(np.sort([np.max(e) for e in rr.tolist()]))
            plt.plot(summaryconfidence, label="Summary confidence")
            #plt.plot(confidence, label="Confidence")
            plt.yticks(np.arange(0, max(summaryconfidence), step=0.2))
            plt.hlines([0.0,0.2,0.4,0.6,0.8,1],0,len(summaryconfidence),linestyles='dotted')
            plt.legend()
            plt.savefig(f"{datadir}/tmp/{dsName}{verNr}.png")

            if len(input_file_test) > 0:
                # Saving intents sorted by confidence for each test example
                output_file = f"{datadir}/tmp/{dsName}{verNr}.txt"
                with open(output_file, "w", encoding="utf-8") as f:
                    for row, e in enumerate(rr.tolist()):
                        topindices = np.flip(np.argsort(e))
                        f.write("{}\t\t".format(test_ex[row]['text']))
                        for item in topindices:
                            f.write("{}\t{}\t\t".format(format(e[item], '.8f'),intents[item]))
                        f.write('\n')   

        else: # Train with x-validation
            #regParam2Arr = [0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
            regParam2Arr = [0]
            regParam = 0
            for epochs in allEp:
                for regParam2 in regParam2Arr:

                    trainX = np.array([ex['X'] for ex in train_ex])
                    trainY = np.array([ex['Y'] for ex in train_ex])

                    label = f"{dsName}_{verNr}_{regParam2}_{epochs}"
                    print("\n" + label, end=" --- ")
                    print(f"train: {len(trainX)}")
                    cl.performXValMultiTest(trainX, trainY, intentmap, label, multi=multi, modelVersion=verNr, regParam=regParam, regParam2=regParam2, epochs=epochs)
            
            # Plotting precision, recall, f1, accuracy for the different tresholds
            predictionArr = []
            startTime = datetime.datetime.now()
            dateText = startTime.strftime("%Y-%m-%d")
            for foldNr in range(5):
                resfile = f"{resdir}/{dateText}/{label}/fold{foldNr}/all_intents/raw_preds.txt"
                with open(resfile, "r", encoding="utf-8") as f:
                    for line in f:
                        lineArr = np.array(json.loads(line))
                        predictionArr.append(lineArr)
            arr = np.vstack(predictionArr)

            trainX = np.array([ex['X'] for ex in train_ex])
            trainY = np.array([ex['Y'] for ex in train_ex])
            examples = np.array([ex['text'] for ex in train_ex])
            ex, ans, y = cl.getXValTestSets(trainX, trainY, examples, intentmap)

            with open(f"{datadir}/tmp/{dsName}{verNr}shuffled.txt","w", encoding="utf-8") as o:
                for idx, currexample in enumerate(ex):
                    o.write("{}\t\t".format(currexample))
                    nonzeroindices = np.where(y[idx] > 0)
                    o.write("{}\t{}\t".format(len(nonzeroindices[0]),'; '.join(np.array(intents)[nonzeroindices])))
                    topindices = np.flip(np.argsort(predictionArr[idx]))
                    for item in topindices:
                        o.write("{}\t{}\t\t".format(format(predictionArr[idx][item], '.8f'),intents[item]))
                    o.write('\n')  
            
            expectedy = np.array(y)
            predictedy = np.array(predictionArr)
            PlotandSave(expectedy, predictedy, f"{datadir}/tmp/{dsName}{verNr}", steps=1001)
            
            num_intents = np.sum(expectedy, axis=1)
            index_0 = np.where(num_intents == 0)
            expected_0 = expectedy[index_0]
            predicted_0 = predictedy[index_0]
            PlotandSave(expected_0, predicted_0, f"{datadir}/tmp/{dsName}{verNr}0", steps=1001)

            index_1 = np.where(num_intents == 1)
            expected_1 = expectedy[index_1]
            predicted_1 = predictedy[index_1]
            PlotandSave(expected_1, predicted_1, f"{datadir}/tmp/{dsName}{verNr}1", steps=1001)

            index_2 = np.where(num_intents >1)
            expected_2 = expectedy[index_2]
            predicted_2 = predictedy[index_2]
            PlotandSave(expected_2, predicted_2, f"{datadir}/tmp/{dsName}{verNr}2more", steps=1001)

    except KeyboardInterrupt:
        sys.stdout.flush()
        pass


if __name__ == '__main__':
    sys.exit(int(main() or 0))