# wer_are_we
WER are we? An attempt at tracking states of the art(s) and recent results on speech recognition. *Feel free to correct!* 
(Inspired by [Are we there yet?](http://rodrigob.github.io/are_we_there_yet/build/))

To be updated with Interspeech 2015...

## WER

### LibriSpeech

(Possibly trained on more data than LibriSpeech.)

| WER test-clean | WER test-other | Paper          | Notes   |
| :------------- | :------------- | :------------- | :-----: |
| 5.83% | 12.69 | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | *Humans* |
| 4.83% | | [A time delay neural network architecture for efficient modeling of long temporal contexts](http://speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) | TDNN + iVectors |
| 5.33% | 13.25% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 68M parameters |
| 5.51% | 13.97% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |
| 8.01% | 22.49% | same, [Kaldi](http://kaldi-asr.org/) | HMM-(SAT)GMM |
| | 12.51% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | TDNN + pNorm + speed up/down speech |


### WSJ

(Possibly trained on more data than WSJ.)

| WER eval'92    | WER eval'93    | Paper          | Notes   |
| :------------- | :------------- | :------------- | :-----: |
| 5.03% | 8.08% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | *Humans* |
| 3.63% | 5.66% |[LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | test-set on open vocabulary (i.e. harder), model = HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |
| 3.60% | 4.98% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 68M parameters |
| 5.6% | | [Convolutional Neural Networks-based Continuous Speech Recognition using Raw Speech Signal](http://infoscience.epfl.ch/record/203464/files/Palaz_Idiap-RR-18-2014.pdf) | CNN over RAW speech (wav) |

### Switchboard Hub5'00

(Possibly trained on more data than SWB, but test set = full Hub5'00.)

| WER (SWB) | WER (full=SWB+CH) | Paper          | Notes   |
| :------------- | :---------------- | :------------- | :-----: |
| 8% | 14.1% | [The IBM 2015 English Conversational Telephone Speech Recognition System](http://arxiv.org/pdf/1505.05899.pdf) | CNN+RNN (lattice-based MBR loss) with maxout + annealed dropout trained on SWB+Fisher+CH. NNLM scoring. |
| 12.6% | 16% | [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567) | CNN + Bi-RNN + CTC (speech to letters), 25.9% WER if trained _only_ on SWB |
| 12.6% | 18.4% | [Sequence-discriminative training of deep neural networks](http://www.danielpovey.com/files/2013_interspeech_dnn.pdf) | HMM-DNN +sMBR |
| 12.9% | 19.3% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | TDNN + pNorm + speed up/down speech |
| 15% | 19.1% | [Building DNN Acoustic Models for Large Vocabulary Speech Recognition](http://arxiv.org/abs/1406.7806v2) | DNN + Dropout |
| 10.4% | | [Joint Training of Convolutional and Non-Convolutional Neural Networks](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p5609-soltau.pdf) | CNN on MFSC/fbanks + 1 non-conv layer for FMLLR/I-Vectors concatenated in a DNN |
| 11.5% | | [Deep Convolutional Neural Networks for LVCSR](http://www.cs.toronto.edu/~asamir/papers/icassp13_cnn.pdf) | CNN |
| 12.2% | | [Very Deep Multilingual Convolutional Neural Networks for LVCSR](http://arxiv.org/pdf/1509.08967v1.pdf) | Deep CNN (10 conv, 4 FC layers), multi-scale feature maps |

### CHiME (noisy speech)

| clean | real | sim | Paper | Notes |
| :------ | :----- | :----- | :----- | :-----: |
| 3.34% | 21.79% | 45.05% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 68M parameters |
| 6.30% | 67.94% | 80.27% | [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567) | CNN + Bi-RNN + CTC (speech to letters) |

TODO

## PER

### TIMIT

(So far, all results trained on TIMIT and tested on the standard test set.)

| PER     | Paper  | Notes   | 
| :------ | :----- | :-----: |
| 16.7%   | [Combining Time- and Frequency-Domain Convolution in Convolutional Neural Network-Based Phone Recognition](http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2014.pdf) | CNN in time and frequency + dropout, 17.6% w/o dropout |
| 17.6%   | [Attention-Based Models for Speech Recognition](http://arxiv.org/abs/1506.07503) | Bi-RNN + Attention |
| 17.7%   | [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/abs/1303.5778v1) | Bi-LSTM + skip connections w/ CTC |
| 23%     | [Deep Belief Networks for Phone Recognition](http://www.cs.toronto.edu/~asamir/papers/NIPS09.pdf) | (first, modern) HMM-DBN |

## LM
TODO

## Noise-robust ASR
TODO

## BigCorp™®-specific dataset
TODO?

## Lexicon
 * WER: word error rate
 * PER: phone error rate
 * LM: language model
 * HMM: hidden markov model
 * GMM: Gaussian mixture model
 * DNN: deep neural network
 * CNN: convolutional neural network
 * DBN: deep belief network (RBM-based DNN)
 * RNN: recurrent neural network
 * LSTM: long short-term memory
 * CTC: connectionist temporal classification
 * MMI: maximum mutual information (MMI),
 * MPE: minimum phone error 
 * sMBR: state-level minimum Bayes risk
 * SAT: speaker adaptive training
 * MLLR: maximum likelihood linear regression
 * LDA: (in this context) linear discriminant analysis
 * MFCC: [Mel frequency cepstral coefficients](http://snippyhollow.github.io/blog/2014/09/25/classical-speech-recognition-features-in-one-picture/)
 * FB/FBANKS/MFSC: [Mel frequency spectral coefficients](http://snippyhollow.github.io/blog/2014/09/25/classical-speech-recognition-features-in-one-picture/) 
