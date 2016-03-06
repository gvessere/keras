# Generate from sequencing data

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file, Progbar
from tempfile import mkstemp
from datetime import datetime
from keras.callbacks import *
import random
import os as os
#import matplotlib.pyplot as plt
import numpy as np
import sys

dropout=0.2
lstm_layers=4
batchsteps=10
batchsize=512
timesteps=200

if len(sys.argv) > 1:
    dropout=float(sys.argv[1])
if len(sys.argv) > 2:
    lstm_layers=int(sys.argv[2])
if len(sys.argv) > 3:
    batchsteps=int(sys.argv[3])

print("dropout(%0.3f), lstm_layers(%d), batchsteps(%d), batchsize(%d), timesteps(%d)" % (dropout, lstm_layers, batchsteps, batchsize, timesteps))

assert lstm_layers >= 1,"need more than one lstm layers"

resultfile,resultfilename=mkstemp(prefix='training.',suffix='.txt', dir=os.getcwd())
tmppart=resultfilename.replace(os.getcwd() + '/', '').replace('training.','').replace('.txt','')

os.write(resultfile,"char-rnn dna model: " + str(datetime.now()) + '\n')
os.write(resultfile,"dropout: %0.3f\n" % (dropout))
os.write(resultfile,"lstmlayers:%d\n" % (lstm_layers))
os.write(resultfile,"batchsteps:%d\n" % (batchsteps))
os.write(resultfile,"batchsize:%d\n" % (batchsize))
os.write(resultfile,"timesteps:%d\n" % (timesteps))

# load sequence data
# input sequence file obtained with the following commands 
# returns all gene regions in separate records with uppercased exons
# wget -O kg_one_record_per_region.fasta "http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=476467603_ZpmULv360T1UveJRDFvf1WjGEDq6&hgSeq.promoter=on&boolshad.hgSeq.promoter=0&hgSeq.promoterSize=300&hgSeq.utrExon5=on&boolshad.hgSeq.utrExon5=0&hgSeq.cdsExon=on&boolshad.hgSeq.cdsExon=0&hgSeq.utrExon3=on&boolshad.hgSeq.utrExon3=0&hgSeq.intron=on&boolshad.hgSeq.intron=0&hgSeq.downstream=on&boolshad.hgSeq.downstream=0&hgSeq.downstreamSize=300&hgSeq.granularity=feature&hgSeq.padding5=50&hgSeq.padding3=50&hgSeq.splitCDSUTR=on&boolshad.hgSeq.splitCDSUTR=0&hgSeq.casing=exon&boolshad.hgSeq.maskRepeats=0&hgSeq.repMasking=lower&hgta_doGenomicDna=get+sequence"
# cat kg_one_record_per_region.fasta | perl -pe "s/^(>.+)$/\1|/g" |perl -pe "s/\|/\t/g" | perl -pe "s/=([-+])/\t\1\t/" | perl -pe "s/chr(.+?):(\d+)-(\d+)/\t\1\t\2\t\3\t/" | cut -f1,2,3,4,6,8 | perl -pe "s/>.+?\t/|/s" | perl -pe "s/\n//s" | tr "|" "\n" | cut -f5 | sort | uniq | grep -v "n" > ~/.keras/datasets/kg_one_record_per_region.seq.uq

path = get_file('kg_one_record_per_region.seq.uq', origin="")

text = open(path).read()
print('corpus length:', len(text))
genes=text.split('\n')
random.seed(2016)
random.shuffle(genes)

print('total sequences:', len(genes))

chars = set(text.replace('\n',''))
print('total chars:', len(chars), chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def getNNData(data, minLength, maxLength):
    sentences = []
    for i in range(len(data)):
        l=len(data[i])
        if minLength <= l and l < maxLength:
            sentences.append(data[i][:-1])
    
    #sentences = sentences[:len(sentences)/5]
    
    # ensure this is a multiple of batchsize
    sentences=sentences[:len(sentences)/batchsize*batchsize]
    print('nb sequences in range(' + str(minLength) + ',' + str(maxLength) + ') : ', len(sentences))
    
    print('Vectorization...')
    X = np.zeros((len(sentences), maxLength, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        # make sure sequence length is a multiple of batchsteps
        # or we will predict no characters on unmasked seq at the end
        sentence=sentence[:1+(len(sentence)-1)/batchsteps*batchsteps]
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
    
    return X


class ResetStates(Callback):
    def __init__(self, timesteps, batchsteps, filesuffix):
        self.timesteps=timesteps
        self.batchsteps=batchsteps
        self.file=open("results." + filesuffix + ".tsv","a")
        
    def on_batch_end(self, batch, logs={}):
        if (batch !=0 and batch % (self.timesteps/self.batchsteps) == 0):
            print()
            print()
            print("new batch group, resetting states")
            self.model.reset_states()
        
            for k, v in logs.items():
                self.file.write("%s\t%s\t" % (k, v))
            self.file.write("\n")
            self.file.flush()

def shuffle_for_stateful(X):
    assert(X.shape[0]%batchsize==0)
    assert(X.shape[1]%batchsteps==0)
    

    Xs=X.reshape(X.shape[0]/batchsize, batchsize, X.shape[1], X.shape[2])
    ys=np.zeros(Xs.shape, dtype=bool)
    ys[:,:,:-batchsteps,:]=Xs[:,:,batchsteps:,:]
    Xs=Xs.reshape(Xs.shape[0], Xs.shape[1], Xs.shape[2]/batchsteps, batchsteps, Xs.shape[3])
    Xs=Xs.transpose(0,2,1,3,4)
    Xs=Xs.reshape(-1, Xs.shape[3], Xs.shape[4])
    ys=ys[:,:,::batchsteps,:]
    ys=ys.transpose(0,2,1,3)
    ys=ys.reshape(-1,ys.shape[3])

    # for i in range(2*batchsize):
    #     data=""
    #     #print(len(Xgrouped[i]))
    #     for j,onehot in enumerate(Xgrouped[i]):
    #         ci=np.where(onehot==True)
    #         if len(ci[0]>0):
    #             c=indices_char[ci[0][0]]
    #         else:
    #             c=" "
    #         data=data+c
    #     ci=np.where(ygrouped[i]==True)
    #     if len(ci[0]>0):
    #         c=indices_char[ci[0][0]]
    #         data=data+"-"+c
    #     print(data)
            
    return (Xs, ys)

 
def predSeq(seq, steps=batchsteps):
    from heapq import nlargest
    x = np.zeros((batchsize, timesteps, len(chars)), dtype=np.bool)
    for t, char in enumerate(seq):
        x[0, t, char_indices[char]] = True
        
    model.reset_states()
    for i in np.arange(0,len(seq)-steps,steps):
        p=model.predict(x[:,i:i+steps,:], batch_size=batchsize)[0]
        indexes=range(p.size)
        preds=nlargest(4,indexes, key=lambda i: p[i])
        print(seq[i+steps], indices_char[preds[0]],indices_char[preds[1]],indices_char[preds[2]],indices_char[preds[3]], p[preds[0]],p[preds[1]],p[preds[2]],p[preds[3]])

print("Preparing training data...")
X=getNNData(genes, 0,timesteps)
validatefraction=.05
valcount=((int) (len(X) * validatefraction)) / batchsize * batchsize
Xval=X[-valcount:]

X=X[:-valcount]
[X, y] = shuffle_for_stateful(X)
[Xval, yval] = shuffle_for_stateful(Xval)


print('Building model...')
model = Sequential()
model.add(Masking(mask_value=0, batch_input_shape=[batchsize,batchsteps,len(chars)]))
for i in range(lstm_layers-1):
    model.add(LSTM(128, return_sequences=True, stateful=True))
    model.add(Dropout(dropout))
model.add(LSTM(128, return_sequences=False, stateful=True))
model.add(Dropout(dropout))
model.add(Dense(len(chars))) # input dim (512,) output dim (8,)
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print('Done')


resetstates=ResetStates(timesteps, batchsteps, tmppart)
cp=ModelCheckpoint('dnaModel3.' + tmppart + '.mod',save_best_only=True)

hist=model.fit(X,y, shuffle=False,validation_data=(Xval,yval), batch_size= batchsize, nb_epoch=100, show_accuracy=True, callbacks = [resetstates, cp])
os.write(resultfile, hist.history)


# # plt.title('training loss')
# # plt.plot(losses)
# # plt.show()


# model.load_weights('dnaModel3.1XK4Qt.mod')
# seq = genes[2]
# predSeq(seq)
