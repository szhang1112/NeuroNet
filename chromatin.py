# -*- coding: utf-8 -*-
"""Compute MN chromatin features (H3K27ac, H3K4me1, H3K4me3, ATAC) of variants (required by predict.py).
Modified from https://github.com/FunctionLab/ExPecto
Example:
        $ python chromatin.py var.vcf var_feat.h5
"""
import argparse
import math
import pyfasta
import torch
from torch import nn
import numpy as np
import pandas as pd
import h5py
import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Predict variant chromatin effects')
parser.add_argument('inputfile', type=str, help='Input file in vcf format')
parser.add_argument('outputfile', type=str, help='Input file in h5 format')
parser.add_argument('--inputsize', action="store", dest="inputsize", type=int,
                    default=2000, help="The input sequence window size for neural network")
parser.add_argument('--batchsize', action="store", dest="batchsize",
                    type=int, default=32, help="Batch size for neural network predictions.")
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
args.cuda = True

genome = pyfasta.Fasta('data/hg19.fa')

CHRS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
        'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX','chrY']

inputfile = args.inputfile
outputfile = args.outputfile
inputsize = args.inputsize
batchSize = args.batchsize

mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
    'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
    'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
    'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
    'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
    'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

def encode(line, inputsize=2000):
    seqsnp = np.zeros((1, 4, inputsize), np.bool_)
    for i, c in enumerate(line):
        seqsnp[0, :, i] = mydict[c]
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp

def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.
    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output
    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize
    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """        
    with Pool(64) as pool:
        seqs = pool.map(encode, seqs)

    return np.concatenate(seqs, 0)

def fetchSeqs(chr, pos, ref, alt, inputsize=2000):
    windowsize = inputsize
    mutpos = int(windowsize / 2 - 1)
    seq = genome.sequence({'chr': chr, 'start': pos -
                           int(windowsize / 2 - 1), 'stop': pos + int(windowsize / 2)})
    return seq[:mutpos] + ref + seq[(mutpos + 1):], seq[:mutpos] + alt + seq[(mutpos + 1):], seq[mutpos:(mutpos + 1)].upper() == ref.upper()


vcf = pd.read_csv(inputfile, sep='\t', header=None, comment='#')

# standardize
vcf.iloc[:, 0] = 'chr' + vcf.iloc[:, 0].map(str).str.replace('chr', '')
vcf = vcf[vcf.iloc[:, 0].isin(CHRS)]
vcf = vcf.reset_index(drop=True)

refseqs = []
altseqs = []
ref_matched_bools = []

for i in range(vcf.shape[0]):
    refseq, altseq, ref_matched_bool = fetchSeqs(
        vcf[0][i], vcf[1][i], vcf[3][i], vcf[4][i], inputsize=inputsize)
    if ref_matched_bool:
        refseqs.append(refseq)
        altseqs.append(altseq)
    ref_matched_bools.append(ref_matched_bool)

print("Number of variants with reference allele matched with reference genome:")
print(np.sum(ref_matched_bools))
print("Number of input variants:")
print(len(ref_matched_bools))

ref_encoded = encodeSeqs(refseqs, inputsize=inputsize).astype(np.float32)
print('done!')
alt_encoded = encodeSeqs(altseqs, inputsize=inputsize).astype(np.float32)
print('finish encode sequence!')

from run4_from_scratch import ChromatinModel

model1 = ChromatinModel().load_from_checkpoint('model/epoch=5-step=51593.ckpt').backbone
model1.eval()
if args.cuda:
    model1.cuda()

for model, prefix in zip([model1], ['four']):
    ref_preds = []
    for i in tqdm.tqdm(range(int(1 + (ref_encoded.shape[0]-1) / batchSize))):
        input = torch.from_numpy(ref_encoded[int(i*batchSize):int((i+1)*batchSize),:,:])
        if args.cuda:
            input = input.cuda()
        ref_preds.append(model.forward(input).cpu().detach().numpy().copy())
    ref_preds = np.vstack(ref_preds)

    alt_preds = []
    for i in range(int(1 + (alt_encoded.shape[0]-1) / batchSize)):
        input = torch.from_numpy(alt_encoded[int(i*batchSize):int((i+1)*batchSize),:,:])
        if args.cuda:
            input = input.cuda()
        alt_preds.append(model.forward(input).cpu().detach().numpy().copy())
    alt_preds = np.vstack(alt_preds)

    f = h5py.File(outputfile, 'w')
    f.create_dataset('ref', data=ref_preds)
    f.create_dataset('alt', data=alt_preds)
    f.create_dataset('flag', data=ref_matched_bools)
    f.close()

