import socket
import datetime
import sys
import codecs
from functools import lru_cache

def ts(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f: ")

class VectorizerConnector(object):
    """For interacting with VectorizerService"""
    def __init__(self, address, port, lru_cache_size=10000):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((address, port))
        self.partial_decoder = codecs.getincrementaldecoder('utf8')()
        self.vectorizeWord = lru_cache(lru_cache_size)(self.vectorizeWord)
        print(ts() + "Successfully connected to the vectorizer")
    #def vectorizeList(self, words):
    #    #TODO: implement
    def sendMessage(self, msg):
        while len(msg) > 0:
            bytes_sent = self.s.send(msg)
            msg = msg[bytes_sent:]
    
    #@lru_cache(10000) # moved to __init__ to allow different param
    def vectorizeWord(self, priority, word):
        #return [0] * 300
        msg = (str(priority) + word + "\n").encode("utf-8")
        self.sendMessage(msg)
        
        doneReceiving = False
        decodedData = ""
        while not doneReceiving:
            raw_data = self.s.recv(1024)
            if raw_data:
                decodedData += self.partial_decoder.decode(raw_data)
            else:
                raise ConnectionAbortedError("The vectorizer service closed the connection!")
            if decodedData[-1] == '\n': # this works only for the current protocol, i.e, when sending strings
                doneReceiving = True
        
        retval = VectorizerConnector.parseOutput(decodedData)
        return retval

    def vectorize(self, utterance, priority=1):
        retval = []
        for word in utterance.split():
            vec = self.vectorizeWord(priority, word)
            retval.append(vec)
        return retval

    def closeConnection(self):
        self.s.close()
        pass

    def parseOutput(data):
        assert data[0] == "["
        assert data[-2:] == "]\n"
        retval = [float(x) for x in data[1:-2].split(", ")]
        return retval
        



