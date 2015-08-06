# wer_are_we
WER are we? An attempt at tracking states of the arts and recent results on speech recognition. *Feel free to correct!* 
(Inspired by [Are we there yet?](http://rodrigob.github.io/are_we_there_yet/build/))

## WER

### LibriSpeech

(Possibly trained on more data than LibriSpeech.)

| WER test-clean | WER test-other | Paper          | Notes   |
| :------------- | :------------- | :------------- | :-----: |
| 5.51% | 13.97% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |


### WSJ

(Possibly trained on more data than WSJ.)

| WER eval'92    | WER eval'93    | Paper          | Notes   |
| :------------- | :------------- | :------------- | :-----: |
| 3.63% | 5.66% |[LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | test-set on open vocabulary (i.e. harder), model = HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |
| TODO | | |

### Switchboard Hub5'00

(Possibly trained on more data than SWB, but test set = full Hub5'00.)

| WER (SWB=easy) | WER (full=SWB+CH) | Paper          | Notes   |
| :------------- | :---------------- | :------------- | :-----: |
| 12.6% | 16% | http://arxiv.org/abs/1412.5567 | CNN + Bi-RNN + CTC (speech to letters), 25.9% WER if trained _only_ only SWB |
| 12.6% | 18.4% | http://www.danielpovey.com/files/2013_interspeech_dnn.pdf | HMM-DNN +sMBR |
| 16% | 19.9% | http://arxiv.org/abs/1406.7806v2 | TODO |
| 10.4% | | http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p5609-soltau.pdf | TODO |
| 11.5% | | http://www.cs.toronto.edu/~asamir/papers/icassp13_cnn.pdf | TODO |

## PER

### TIMIT

(So far, all results trained on TIMIT and tested on the standard test set.)

| PER     | Paper  | Notes   | 
| :------ | :----- | :-----: |
| 16.7%   | http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2014.pdf | CNN in time and frequency + dropout, 17.6% w/o dropout |
| 17.6%   | http://arxiv.org/abs/1506.07503 | Bi-RNN + Attention |
| 17.7%   | http://arxiv.org/abs/1303.5778v1 | Bi-LSTM + skip connections w/ CTC |
| 23%     | http://www.cs.toronto.edu/~asamir/papers/NIPS09.pdf | HMM-DBN |

## LM
TODO

## Noise-robust ASR
TODO

## BigCorp™®-specific dataset
TODO?

## Lexicon
 * WER: word error rate
 * PER: whone error rate
 * LM: language model
 * HMM: hidden markov model
 * GMM: Gaussian mixture model
 * DNN: deep neural network
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
