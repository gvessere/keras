from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys
import theano


maxlen=60
path=get_file("genes.small.txt",origin="http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=470093087_UHcfeHatgzQoJLFCwCJC3C3wTVav&boolshad.hgSeq.promoter=0&hgSeq.promoterSize=1000&hgSeq.utrExon5=on&boolshad.hgSeq.utrExon5=0&hgSeq.cdsExon=on&boolshad.hgSeq.cdsExon=0&hgSeq.utrExon3=on&boolshad.hgSeq.utrExon3=0&hgSeq.intron=on&boolshad.hgSeq.intron=0&boolshad.hgSeq.downstream=0&hgSeq.downstreamSize=1000&hgSeq.granularity=gene&hgSeq.padding5=0&hgSeq.padding3=0&boolshad.hgSeq.splitCDSUTR=0&hgSeq.casing=exon&boolshad.hgSeq.maskRepeats=0&hgSeq.repMasking=lower&hgta_doGenomicDna=get+sequence")
text=open(path).read()
chars=set(text)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def generate(sentence, length):
    generated=""
    for i in range(length):
        x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.
    
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]
    
    generated += next_char
    sentence = sentence[1:] + next_char
    
    sys.stdout.write(next_char)
    sys.stdout.flush()
    return generated


def buildModel():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True,
    stateful=True,
    batch_input_shape=(1,maxlen, len(chars))))
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False,
    stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    print("compiling model")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    print("done")
    return model


model=buildModel()
model.load_weights('dnaModel.mod')

diversity=1.0

#print("trying to generate some random sequence to see if it's decent")
#generate(">hg38_knownGene_uc031tla.1 range=chr1:17369-17436 5'pad=0 3'pad=0 strand=- repeatMasking=none"[:maxlen], 100)

print("trying to repro results from The ""Unreasonable Effectiveness of Recurrent Neural Networks"" (karpathy.github.io/2015/05/21/rnn-effectiveness/). Will try to see what structures it found on the genes (other than the obvious intron-exon junctions)")

print("feeding an existing sequence to the trained dna model")
dna = """>hg38_knownGene_uc057atz.1 range=chr1:30267-31109 5'pad=0 3'pad=0 strand=+ repeatMasking=none
TCATCAGTCCAAAGTCCAGCAGTTGTCCCTCCTGGAATCCGTTGGCTTGC
CTCCGGCATTTTTGGCCCTTGCCTTTTAGGGTTGCCAGATTAAAAGACAG
GATGCCCAGCTAGTTTGAATTTTAGATAAACAACGAATAATTTCGTAGCA
TAAATATGTCCCAAGCTTAGTTTGGGACATACTTATGCTAAAAAACATTA
TTGGTTGTTTATCTGAGATTCAGAATTAAGCATTTTATATTTTATTTGCT
GCCTCTGGCCACCCTACTCTCTTCCTAACACTCTCTCCCTCTCCCAGTTT
TGTCCGCCTTCCCTGCCTCCTCTTCTGGGGGAGTTAGATCGAGTTGTAAC
AAGAACATGCCACTGTCTCGCTGGCTGCAGCGTGTGGTCCCCTTACCAGA
Ggtaaagaagagatggatctccactcatgttgtagacagaatgtttatgt
cctctccaaatgcttatgttgaaaccctaacccctaatgtgatggtatgt
ggagatgggcctttggtaggtaattacggttagatgaggtcatggggtgg
ggccctcattatagatctggtaagaaaagagagcattgtctctgtgtctc
cctctctctctctctctctctctctcatttctctctatctcatttctctc
tctctcgctatctcatttttctctctctctctttctctcctctgtctttt
cccaccaagTGAGGATGCGAAGAGAAGGTGGCTGTCTGCAAACCAGGAAG
AGAGCCCTCACCGGGAACCCGTCCAGCTGCCACCTTGAACTTGGACTTCC
AAGCCTCCAGAACTGTGAGGGATAAATGTATGATTTTAAAGTC"""

print(dna)
X = np.zeros((1, len(dna), len(chars)))
for t, char in enumerate(dna):
    X[0, t, char_indices[char]] = 1.

print(model.predict(X))

#model.layers[0].input=np.asarray(X, dtype=theano.config.floatX)

#outputs = model.get_output()

#print(model.layers[0].states)

from heapq import nlargest
for i in range(400):
        p=model.predict(X[:,i:60+i,:])
        #model.reset_states()
        indexes=range(p.size)
        preds=nlargest(4,indexes, key=lambda i: p[:,i])
        print(dna[i+60], indices_char[preds[0]],indices_char[preds[1]],indices_char[preds[2]],indices_char[preds[3]], p[:,preds[0]],p[:,preds[1]],p[:,preds[2]],p[:,preds[3]])
