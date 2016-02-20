'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# file with the following commands 
# wget -O dna "http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=476317005_ucqyJHBYfqmTY6NouE8av4h0jaDj&hgSeq.promoter=on&boolshad.hgSeq.promoter=0&hgSeq.promoterSize=1000&hgSeq.utrExon5=on&boolshad.hgSeq.utrExon5=0&hgSeq.cdsExon=on&boolshad.hgSeq.cdsExon=0&hgSeq.utrExon3=on&boolshad.hgSeq.utrExon3=0&hgSeq.intron=on&boolshad.hgSeq.intron=0&hgSeq.downstream=on&boolshad.hgSeq.downstream=0&hgSeq.downstreamSize=1000&hgSeq.granularity=gene&hgSeq.padding5=0&hgSeq.padding3=0&boolshad.hgSeq.splitCDSUTR=0&hgSeq.casing=exon&boolshad.hgSeq.maskRepeats=0&hgSeq.repMasking=lower&hgta_doGenomicDna=get+sequence" 
# cat dna | perl -pe "s/^(>.+)$/\1|/g" |perl -pe "s/\|/\t/g" | perl -pe "s/=([-+])/\t\1\t/" | perl -pe "s/chr(.+?):(\d+)-(\d+)/\t\1\t\2\t\3\t/" | cut -f1,2,3,4,6,8 | perl -pe "s/>.+?\t/|/s" | perl -pe "s/\n//s" | tr "|" "\n" > kg.fasta.formatted.seq.sample


path = get_file('kg.fasta.formatted.seq.sample', origin="")

text = open(path).read()
print('corpus length:', len(text))
genes=text.split('\n')
print('total sequences:', len(genes))

for i in range(len(genes)):
    genes[i]=genes[i] + '.' 
chars = set(genes[1])
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# vectorize genes of size at most maxlen
maxlen = 800 
step = 1
batchsize=32
dropout=0.2

if len(sys.argv) > 1:
    dropout=sys.argv[1]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def getNNData(sentences, minLength, maxLength):
    sentences = []
    next_chars = []
    for i in range(len(genes)):
        if (len(genes[i]) < maxlen) and (len(genes[i]) < maxlen - 200):
            sentences.append(genes[i][:-1])
            next_chars.append(genes[1:i])
        
    print('nb sequences of length at most ' + str(maxlen) + ':', len(sentences))
    #print(sentences)
    print('Vectorization...')
    X = np.zeros((samples, maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((samples, maxlen, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
            
    return (X,y)

def saveModel(model):
    json_string = model.to_json()
    open('dnaModel2.json', 'w').write(json_string)
    model.save_weights('dnaModel2.mod', overwrite=True)    

# build the model: 3 stacked LSTM
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

samples=len(sentences)
trainingSamples=samples*0.9
validationSamples=samples*0.1

# train on data ranges of increasingly long sequences
# eg. 0-200, 200-400 etc...
windowSize = 200
for windowStart in range(0,1000,windowSize):
    (Xtrain,ytrain)=getNNData(sentences[:trainingSamples], windowStart,windowStart + windowSize)
    (Xval,yval)=getNNData(sentences[-validationSamples:], windowStart,windowStart + windowSize)

    # train the model, output generated text after each batch
    losses = []
    for epoch in range(60):
        print("epoch: %d" % epoch)
        for batch in range((samples-1)/batchsize):
            print()
            print('-' * 50)
            for timestep in range(maxlen):
                batchSamples=slice(batchsize*batch,batchsize*(batch+1))
                Xbatch=np.reshape(X[batchSamples,timestep,:], (batchsize, 1, len(chars)))
                ybatch=y[batchSamples,timestep,:]
                loss=model.train_on_batch(Xbatch,ybatch)
                losses.append(loss[0].tolist())
                print("batch %d, timestep %d, loss: %0.3f" % (batch, timestep, losses[-1]), end='\r')
                if timestep == 1:
                    print()
                sys.stdout.flush()
                
            model.reset_states()
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
