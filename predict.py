# -*- coding: utf-8 -*-
"""Predict noncoding variant effect based on pretrained NeuroNet model.
Example:
        $ python predict.py data/example_feat.h5 data/example.vcf data/example_effect.txt
"""
import argparse
import h5py
import numpy as np
from scipy import stats
import pandas as pd

parser = argparse.ArgumentParser(description='Predict noncoding variant effect')
parser.add_argument('inputfile', type=str, help='Input file in h5 format')
parser.add_argument('inputvcf', type=str, help='Input variant file in vcf format')
parser.add_argument('outputfile', type=str, help='Input file in txt format')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
args.cuda = True

four_file = h5py.File(args.inputfile, 'r')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

four_alt = sigmoid(four_file['alt'][:])
four_ref = sigmoid(four_file['ref'][:])

diff = (four_ref - four_alt)
diff_avrg = (diff[1::2] + diff[::2])/2

var = pd.read_csv(args.inputvcf, sep='\t', header=None)

var['h3k27ac'] = np.abs(diff_avrg[:,0])
var['h3k4me1'] = np.abs(diff_avrg[:,1])
var['h3k4me3'] = np.abs(diff_avrg[:,2])
var['atac'] =np.abs(diff_avrg[:,3])
var['max'] = var[['h3k27ac', 'h3k4me1', 'h3k4me3', 'atac']].max(axis=1)
var.columns = ['chr', 'pos', 'variant_id', 'ref', 'alt', 'h3k27ac', 'h3k4me1', 'h3k4me3', 'atac', 'max']

var.to_csv(args.outputfile, sep='\t', index=False)



