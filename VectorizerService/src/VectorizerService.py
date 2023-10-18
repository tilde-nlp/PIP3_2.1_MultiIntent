import sys
import os
import numpy as np
from itertools import chain
import argparse
import datetime
import subprocess
import re
import json

import threading
import socket
from time import sleep
import itertools

import select, socket, sys, queue
from enum import IntEnum
import codecs

#with open("appsettings.json") as f:
#    appsettings = json.load(f)
#logFile = appsettings["ConnectionStrings"]["TempFolder"] + "/VectorizerServicePython.log"

logFile = "./VectorizerServicePython.log"
port = 0
ftModel = ""

totalCycles = 0
totalWordsVectorized = 0
inputs = []
vectorizerQueue = queue.PriorityQueue()
counter = itertools.count()

def ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f: ")

def log(msg, totC=None, toFile=False):
    if totC is not None:
        totC = " (c: {})".format(totC)
        msg += totC
    prefix = "({}, {}): ".format(port, ftModel)
    print(ts() + prefix + msg, flush=True)
    if toFile:
        with open(logFile, 'a', encoding='utf-8') as f:
            print(ts() + prefix + msg, file=f, flush=True)

def toLowercaseCharsTokenized(line):
    # \W = non alphanumeric chars
    line = re.sub(r'[\W_]+', ' ', line, flags=re.UNICODE)
    line = re.sub(r' +', ' ', line, flags=re.UNICODE)
    line = line.strip().lower()
    return line



class FTHandler():
    def __init__(self, ftExec, ftModel, errFile):
        self.ft = subprocess.Popen([ftExec, "print-sentence-vectors", ftModel],
                        stdin=subprocess.PIPE,
                        bufsize=0,
                        encoding='utf-8',
                        stdout=subprocess.PIPE,
                        #stderr=subprocess.PIPE)
                        stderr=errFile)

    def getSentenceVector(self, question):
        question = toLowercaseCharsTokenized(question) + "\n"
        self.ft.stdin.write(question)
        out = self.ft.stdout.readline()
        return [float(x) for x in out.split()]

    def getWordVectors(self, question):
        question = toLowercaseCharsTokenized(question)
        retval = []
        for word in question.split():
            self.ft.stdin.write(word + "\n")
            out = self.ft.stdout.readline()
            retval.append([float(x) for x in out.split()])
        return retval
    
class FakeFTHandler():
    def __init__(self, dim):
        self.dim = dim
        pass

    @staticmethod
    def getSimpleVector(dim, question):
        retval = []
        for i, c in enumerate(question):
            if i > dim:
                break
            retval.append(0.01 * ord(c) - 1)
        if len(retval) < dim:
            retval += [0] * (dim - len(retval))
        return retval
        

    def getSentenceVector(self, question):
        question = toLowercaseCharsTokenized(question) + "\n"
        return FakeFTHandler.getSimpleVector(self.dim, question)

    def getWordVectors(self, question):
        question = toLowercaseCharsTokenized(question)
        retval = []
        for word in question.split():
            retval.append(FakeFTHandler.getSimpleVector(self.dim, question))
        return retval



# 
# https://pymotw.com/3/select/index.html
# https://steelkiwi.com/blog/working-tcp-sockets/
# 
# https://docs.python.org/3.7/howto/sockets.html
# https://docs.python.org/3/library/socket.html
# https://docs.python.org/3/library/ipc.html
# 
# https://stackabuse.com/basic-socket-programming-in-python/
# 

# [ ] some binary format (protocol) instead of string
# 

class Actions(IntEnum):
    VECTORIZE_HIGH_PR = 1
    VECTORIZE_LOW_PR = 2


def report_status():
    while True:
        log("info: {} input connections, {} words to vectorize in queue, {} cycles done, {} words vectorized".format(len(inputs) - 1, vectorizerQueue.qsize(), totalCycles, totalWordsVectorized), toFile=True)
        sleep(60)

def extract_priority(w, connection):
    count = next(counter) # to prevent comparing further -- connections / words
    if w[0] == '1':
        return (Actions.VECTORIZE_HIGH_PR, count, connection, w[1:])
    if w[0] == '2':
        return (Actions.VECTORIZE_LOW_PR, count, connection, w[1:])
    raise ValueError("Bad priority '{}'! Priority should be '1' or '2'.".format(w[0]))

class ConnectionHandler:
    MAX_QUEUE_LEN = 100
    MAX_OUTPUT_QUEUE_LEN = 100
    def __init__(self, connection):
        self.connection = connection
        self.partial_decoder = codecs.getincrementaldecoder('utf8')()
        self.input_string_stream = "" # TODO: maybe more effective data structure than string
        self.is_new = True
        #
        # arr = [1.23, 123, 4234]
        # struct.pack('!{}f'.format(len(arr)), *arr)
        # https://docs.python.org/3/library/struct.html
        # http://stupidpythonideas.blogspot.com/2013/05/sockets-are-byte-streams-not-message.html
        #
        #self.words_queue = queue.Queue()
        self.word_count = 0
        self.outputs_pending = queue.Queue()
        self.has_finalized = False
        self.out_buffer = b''
        pass
    def add_output(self, output):
        self.outputs_pending.put(output)
    def remove_word(self):
        self.word_count -= 1
        pass
    def stream_decode(self, raw_data):
        s = self.partial_decoder.decode(raw_data)
        self.input_string_stream += s
        return self._split_into_words()
    def _split_into_words(self):
        retval = self.input_string_stream.split("\n")
        self.input_string_stream = retval.pop() # leave the last (possibly unfinished) word in stream
        retval = [extract_priority(w, self.connection) for w in retval]
        self.word_count += len(retval)
        if self.word_count > 0:
            self.is_new = False
        return retval
    def is_full(self):
        #return self.words_queue.qsize() >= ConnectionHandler.MAX_QUEUE_LEN
        if self.word_count >= ConnectionHandler.MAX_QUEUE_LEN:
            return True
        if self.outputs_pending.qsize() > ConnectionHandler.MAX_OUTPUT_QUEUE_LEN:
            return True
        return False
    def finalize(self):
        self.has_finalized = True
    def can_remove(self):
        if (self.has_finalized and self.outputs_pending.qsize() == 0
          and self.word_count == 0 and len(self.out_buffer) == 0):
            return True
        return False
    def _try_sending_buffer(self):
        #log("trying to send buffer")
        bytes_sent = self.connection.send(self.out_buffer)
        #log("sent {}/{} bytes to {} of {}".format(bytes_sent, len(self.out_buffer), self.connection.getpeername(), self.out_buffer))
        self.out_buffer = self.out_buffer[bytes_sent:]
    def has_something_to_send(self):
        if len(self.out_buffer) > 0:
            return True
        if self.outputs_pending.qsize() > 0:
            return True
        return False
    def try_sending(self):
        if len(self.out_buffer) > 0:
            self._try_sending_buffer()
        while len(self.out_buffer) == 0 and self.outputs_pending.qsize() > 0:
            item = self.outputs_pending.get_nowait()
            self.out_buffer += (str(item) + "\n").encode('utf-8')
            self._try_sending_buffer()


def doVectorizerServiceLoop(ftHandler, port):
    global totalCycles, totalWordsVectorized, inputs, vectorizerQueue

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # possibly works in more scenarios: server.bind(('', port))
    # server.bind((socket.gethostname(), port))
    server.bind(('', port))
    server.setblocking(False)

    server.listen(5)
    inputs.append(server)
    outputs = []

    connection_handlers = {}

    log("server started")
    
    while inputs:
        # for debugging
        #sleep(0.1)
        connections_to_remove = []
        sleep(0.001)

        totalCycles += 1
         # finds all sockets ready to exchange information
        readable, writable, exceptional = select.select(
            inputs, outputs, inputs, 0)
        for s in readable:
            # server has a new client connection pending
            if s is server:
#                log("r1", totalCycles)
                connection, client_address = s.accept()
                log('new connection: {}, address: {}\n{}'.format(connection.getpeername(), client_address, connection))
                connection.setblocking(0)
                inputs.append(connection)

                connection_handlers[connection] = ConnectionHandler(connection)
            else:
 #               log("r2", totalCycles)
                # some data incoming from a client connection
                handler = connection_handlers[s]
                #if this client's messages queue is full already, don't read
                if handler.is_full():
                    #log('data from {} not read as its queue is already full'.format(s.getpeername()))
                    continue
                try:
                    raw_data = s.recv(1024)
                except ConnectionResetError as e:
                    log("ERROR: Got connection error!", totalCycles, toFile=True)
                    log(str(e), toFile=True)
                    # log("Will remove {}".format(s.getpeername()), toFile=True)
                    log("Will remove the connection", toFile=True)
                    if s not in connections_to_remove:
                        connections_to_remove.append(s)
                    continue
                if raw_data:

                    #log('from {}: received {!r}'.format(s.getpeername(), raw_data))
                    try:
                        msg_list = handler.stream_decode(raw_data)
                    except ValueError as e:
                        if handler.is_new:
                            # assume that this is an HTTP request and get done with it
                            log("Responding to ping")
                            response = 'HTTP/1.0 200 OK\n\nHello World'
                            s.sendall(response.encode())
                            s.close()
                            connections_to_remove.append(s)
                            continue
                        else:
                            log("ERROR: Got error!", totalCycles, toFile=True)
                            log(str(e), toFile=True)
                            log("Will remove {}".format(s.getpeername()), toFile=True)
                            if s not in connections_to_remove:
                                connections_to_remove.append(s)
                            continue
                    except (UnicodeDecodeError, IndexError) as e:
                        log("ERROR: Got error!", totalCycles, toFile=True)
                        log(str(e), toFile=True)
                        log("Will remove {}".format(s.getpeername()), toFile=True)
                        if s not in connections_to_remove:
                            connections_to_remove.append(s)
                        continue
                    
                    for msg in msg_list:
                        vectorizerQueue.put(msg)

                    if s not in outputs:
                        outputs.append(s)
                else:
                    log('{} finalized input'.format(s.getpeername()))
                    handler.finalize()
                    
                    if handler.can_remove():
                        log('{} is done after it closed the connection'.format(s.getpeername()))
                        connections_to_remove.append(s)
                    

        for s in writable:
            handler = connection_handlers[s]
            if handler.has_something_to_send():
                #log('can send something to {}'.format(s.getpeername()))
                handler.try_sending()
            
                if handler.can_remove():
                    log('{} is done after the output was sent'.format(s.getpeername()))
                    if s not in connections_to_remove:
                        connections_to_remove.append(s)

        for s in exceptional:
            log('ERROR: {} is exceptional, therefore will remove it'.format(s.getpeername()))
            if s not in connections_to_remove:
                connections_to_remove.append(s)
            
        for s in connections_to_remove:
            #log('removing {}'.format(s.getpeername()))
            log('removing some connection')
            if s in inputs:
                inputs.remove(s)
            if s in outputs:
                outputs.remove(s)
            s.close()
            del connection_handlers[s]


        # handle the word queue
        MAX_WORDS_TO_HANDLE = 1
        words_handled = 0
        while vectorizerQueue.qsize() > 0:
            priority, cnt, connection, word = vectorizerQueue.get_nowait()
            #TODO: possibly could fail here - if there are some words to vectorize but the connection has been already closed
            handler = connection_handlers[connection]
            handler.remove_word()
            # log('vectorizing for {} word (no. {}) "{}" (priority: {}), cycle: {}'.format(connection.getpeername(), cnt, word, priority, totalCycles))
            vectorized_word = ftHandler.getSentenceVector(word)
            totalWordsVectorized += 1
            handler.add_output(vectorized_word)
            
            words_handled += 1
            if words_handled >= MAX_WORDS_TO_HANDLE:
                break
        
    pass



def runVectorizer(ftExec, ftModel, port):
    log("----------------------------------------------")
    log("Using " + ftModel)
    errFile = open("./VectorizerErrors.log", "a", buffering=1, encoding="utf-8")
    if ftExec == "fakeFT":
        log(f"Starting fake with dim {int(ftModel)}")
        ftHandler = FakeFTHandler(dim=int(ftModel))
    elif ftModel == "fakeFT":
        log(f"Starting fake with default dim 300")
        ftHandler = FakeFTHandler(dim=300)
    else:
        ftHandler = FTHandler(ftExec, ftModel, errFile)
    ftHandler.getWordVectors("a")
    ftHandler.getWordVectors("b")
    log("Model loaded")

    log("Starting service on port {}".format(port))

    t = threading.Thread(target=report_status)
    t.daemon = True
    t.start()
    
    doVectorizerServiceLoop(ftHandler, port)

    log("Finished")

    pass


def main():
    global port, ftModel
    # params: ../../data/ft/fasttext.exe ../../data/ft/mono.std.def.bin

    parser = argparse.ArgumentParser()
    parser.add_argument('fastText', help='fastText executable path')
    parser.add_argument('model', help='fastText model file path')
    parser.add_argument('port', help='listening port', nargs='?', default=12345)
    args = parser.parse_args()
 
    ftExec = args.fastText
    ftModel = args.model
    port = int(args.port)
    
    runVectorizer(ftExec, ftModel, port)

    

if __name__ == "__main__":
    sys.exit(int(main() or 0))

