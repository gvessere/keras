# Generate from sequencing data

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file, Progbar
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# input sequence file obtained with the following commands 
# returns genes with uppercased exons (no 5' and 3' UTRs)
# wget "http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=476467603_ZpmULv360T1UveJRDFvf1WjGEDq6&hgSeq.promoter=on&boolshad.hgSeq.promoter=0&hgSeq.promoterSize=300&hgSeq.utrExon5=on&boolshad.hgSeq.utrExon5=0&hgSeq.cdsExon=on&boolshad.hgSeq.cdsExon=0&hgSeq.utrExon3=on&boolshad.hgSeq.utrExon3=0&hgSeq.intron=on&boolshad.hgSeq.intron=0&hgSeq.downstream=on&boolshad.hgSeq.downstream=0&hgSeq.downstreamSize=300&hgSeq.granularity=feature&hgSeq.padding5=50&hgSeq.padding3=50&hgSeq.splitCDSUTR=on&boolshad.hgSeq.splitCDSUTR=0&hgSeq.casing=exon&boolshad.hgSeq.maskRepeats=0&hgSeq.repMasking=lower&hgta_doGenomicDna=get+sequence" -O kg_one_record_per_region.fasta
# cat kg_one_record_per_region.fasta | perl -pe "s/^(>.+)$/\1|/g" |perl -pe "s/\|/\t/g" | perl -pe "s/=([-+])/\t\1\t/" | perl -pe "s/chr(.+?):(\d+)-(\d+)/\t\1\t\2\t\3\t/" | cut -f1,2,3,4,6,8 | perl -pe "s/>.+?\t/|/s" | perl -pe "s/\n//s" | tr "|" "\n" | cut -f5 | sort | uniq > ~/.keras/datasets/kg_one_record_per_region.seq.uq

path = get_file('kg_one_record_per_region.seq.uq', origin="")

text = open(path).read()
print('corpus length:', len(text))
genes=text.split('\n')
print('total sequences:', len(genes))

chars = set(genes[2])
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# vectorize genes of size at most maxlen
maxlen = 800 
step = 1
batchsize=256
dropout=0.2

if len(sys.argv) > 1:
    dropout=sys.argv[1]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def getNNData(genes, minLength, maxLength):
    sentences = []
    for i in range(len(genes)):
        l=len(genes[i])
        if minLength <= l and l < maxLength:
            sentences.append(genes[i][:-1])
        
    print('nb sequences in range(' + str(minLength) + ',' + str(maxLength) + ') : ', len(sentences))
    #print(sentences)
    print('Vectorization...')
    X = np.zeros((len(sentences), maxLength, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
    
    y=np.zeros(X.shape)
    y[:,:-1,:]=X[:,1:,:]
    return (X,y)

def saveModel(model):
    json_string = model.to_json()
    open('dnaModel2.json', 'w').write(json_string)
    model.save_weights('dnaModel2.mod', overwrite=True)    

def createBatch(batchId, batchsize, X,y):
    slicemin = batchsize*batchId
    slicemax = min(batchsize*(batchId+1), X.shape[0])
    sliceSize = slicemax-slicemin
    batchSamples=slice(slicemin, slicemax)
    Xbatch=np.zeros((sliceSize,1,len(chars)), dtype=np.bool)
    ybatch=np.zeros((sliceSize,len(chars)), dtype=np.bool)
    
    Xbatch[:sliceSize,:,:]=np.reshape(X[batchSamples,timestep,:], (sliceSize, 1, len(chars)))
    ybatch[:sliceSize,:]=y[batchSamples,timestep,:]
    return (Xbatch,ybatch)
    

def doValidation():
    model.save_weights('validation.mod', overwrite=True)
    model.reset_states()

    for batch in range(len(Xval)/batchsize):
        for timestep in range(windowEnd):
            Xbatch,ybatch=createBatch(batch, batchsize, Xval, yval)
            nonmasked = np.sum(Xbatch)
            
            if nonmasked > 0:
                [ loss, accuracy ] = model.train_on_batch(Xbatch,ybatch, accuracy=True)
                normloss = losses[-1][0].tolist / nonmasked
                print("val loss: %0.3f accuracy: %0.3f, normalized loss: %0.3f" % (loss, accuracy, normloss))
            
    model.load_weights('validation.mod') 

print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, batch_input_shape=[batchsize,1,len(chars)]))
model.add(LSTM(512, return_sequences=True, stateful=True))
model.add(Dropout(dropout))
model.add(LSTM(512, return_sequences=False, stateful=True))
model.add(Dropout(dropout))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

samples=len(genes)
trainingSamples=int(samples*0.8)
validationSamples=int(samples*0.2)

# train on data ranges of increasingly long sequences
# eg. 0-200, 200-400 etc...
windowSize = 200

    # train the model, output generated text after each batch
losses = []
for epoch in range(60):
    print("")
    print("epoch: %d" % epoch)
    for windowStart in range(0,1600,windowSize):
        windowEnd=windowStart+windowSize
        print("")
        print("processing sequences in window %d-%d" % (windowStart, windowEnd))
        (Xtrain,ytrain)=getNNData(genes[:trainingSamples], windowStart,windowEnd)
        (Xval,yval)=getNNData(genes[-validationSamples:], windowStart,windowEnd)
        if len(Xtrain) == 0:
            continue
            
        for batch in range(len(Xtrain)/batchsize):
            print()
            print('-' * 50)
            model.reset_states()
            progbar=Progbar(windowEnd)
            for timestep in range(windowEnd):
                Xbatch,ybatch=createBatch(batch, batchsize, Xtrain, ytrain)
                
                nonmasked = np.sum(Xbatch)
                if (nonmasked > 0):
                    loss=model.train_on_batch(Xbatch,ybatch)
                    losses.append(loss[0].tolist())
                    normloss = losses[-1] / nonmasked
                    progbar.update(1+timestep, values=[('train loss', loss[0].tolist()), ('timestep', timestep), ('sum(nonmasked)', nonmasked), ('normlossx1000', 1000*normloss)])

    doValidation()
    saveModel(model)

    # plt.title('training loss')
    # plt.plot(losses)
    # plt.show()
    
    # start_index = random.randint(0, len(text) - maxlen - 1)

    # for diversity in [0.2, 0.5, 1.0, 1.2]:
    #     print()
    #     print('----- diversity:', diversity)

    #     generated = ''
    #     sentence = text[start_index: start_index + maxlen]
    #     generated += sentence
    #     print('----- Generating with seed: "' + sentence + '"')
    #     sys.stdout.write(generated)

    #     for i in range(400):
    #         x = np.zeros((1, maxlen, len(chars)))
    #         for t, char in enumerate(sentence):
    #             x[0, t, char_indices[char]] = 1.

    #         preds = model.predict(x, verbose=0)[0]
    #         next_index = sample(preds, diversity)
    #         next_char = indices_char[next_index]

    #         generated += next_char
    #         sentence = sentence[1:] + next_char

    #         sys.stdout.write(next_char)
    #         sys.stdout.flush()
    #     print()
