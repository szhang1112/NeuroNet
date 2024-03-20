# NeuroNet: A genomic deep learning model for predicting noncoding variant effect in human motor neurons
NeuroNet is a sequence-based deep learning model which predicts the variant effect on altering epigenomic features (chromatin accessibility and histone marks) in human motor neurons (MNs). NeuroNet follows the Beluga architecture [1] and it was trained based on the MN epigenomic profiling generated in our previous study [2]. More details can be found in our manuscript [3]. Part of our code was modified from https://github.com/FunctionLab/ExPecto.
## 1. How to use NeuroNet?
To use NeuroNet to predict variant effect for your own data, please follow these steps:

1. Download the pretrained NeuroNet model from https://doi.org/10.6084/m9.figshare.25444534 and put it in ./model/.
2. Download the reference genome file (hg19.fa) to ./data/.
3. Create your own variant file (hg19) following the format of ./data/example.vcf.
4. Predict epigenomic features by running
   ```
   python chromatin.py your_own.vcf your_own_feat.h5
   ```
5. Predict variant effect by running
   ```
   python predict.py your_own_feat.h5 your_own.vcf your_own_output.txt
   ```
The code has been tested with Python 3.9.5 and PyTorch 1.8.1. GPU is the default setting.
## 2. Precomputed variant scores
Precomputed variant effect scores used in our manuscript [3] can be downloaded at https://doi.org/10.6084/m9.figshare.25444534.
## 3. References
[1] Zhou, J. et al. Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk. *Nat Genet* **50**, 1171–1179 (2018).

[2] Zhang, S. et al. Genome-wide identification of the genetic basis of amyotrophic lateral sclerosis. *Neuron* **110**, 992–1008.e11 (2022).

[3] Zhang, S. et al. 
