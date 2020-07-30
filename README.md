# wer_are_we
WER are we? An attempt at tracking states of the art(s) and recent results on speech recognition. *Feel free to correct!*
(Inspired by [Are we there yet?](http://rodrigob.github.io/are_we_there_yet/build/))

## WER

### LibriSpeech

(Possibly trained on more data than LibriSpeech.)

| WER test-clean | WER test-other | Paper          | Published | Notes   |
| :------------- | :------------- | :------------- | :-------- | :-----: |
| 5.83% | 12.69% | *Humans* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | December 2015 | *Humans* |
| 1.9%  | 4.1%   | [ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context](https://arxiv.org/pdf/2005.03191.pdf) | May 2020 | CNN-RNN-Transducer(ContextNet) + 3-layer LSTM LM (data augmentation:SpecAugment)  |
| 2.0%  | 4.1%   | [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460) | November 2019 | Conv+Transformer AM (10k word pieces) with ConvLM decoding and Transformer rescoring + 60k hours unlabeled |
| 2.3% | 4.9%  | [Transformer-based Acoustic Modeling for Hybrid Speech Recognition](https://arxiv.org/abs/1910.09799)  | October 2019 | Transformer AM (chenones) + 4-gram LM + Neural LM rescore (data augmentation:Speed perturbation and SpecAugment) |
| 2.3%  | 5.0%   | [RWTH ASR Systems for LibriSpeech: Hybrid vs Attention](https://arxiv.org/abs/1905.03072) | September 2019, Interspeech | HMM-DNN + lattice-based sMBR + LSTM LM + Transformer LM rescoring (no data augmentation) |
| 2.3%  | 5.2%   | [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460) | November 2019 | Conv+Transformer AM (10k word pieces) with ConvLM decoding and Transformer rescoring |
| 2.2%  | 5.8%   | [State-of-the-Art Speech Recognition Using Multi-Stream Self-Attention With Dilated 1D Convolutions](https://arxiv.org/abs/1910.00716) | October 2019 | Multi-stream self-attention in hybrid ASR + 4-gram LM + Neural LM rescore (no data augmentation) |
| 2.5%  | 5.8%   | [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) | April 2019 | Listen Attend Spell |
| 3.2%  | 7.6%  | [From Senones to Chenones: Tied Context-Dependent Graphemes for Hybrid Speech Recognition](https://arxiv.org/abs/1910.01493) | October 2019 | LC-BLSTM AM (chenones) + 4-gram LM (data augmentation:Speed perturbation and SpecAugment) |
| 3.19% | 7.64% | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | April 2018 | TDNN + TDNN-LSTM + CNN-bLSTM + Dense TDNN-LSTM across two kinds of trees + N-gram LM + Neural LM rescore
| 2.44%  |  8.29%   | [Improved Vocal Tract Length Perturbation for a State-of-the-Art End-to-End Speech Recognition System](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/3227.html) | September 2019, Interspeech | encoder-attention-decoder + Transformer LM |
| 3.80% | 8.76%  | [Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks](http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) | Interspeech, Sept 2018 |[Kaldi recipe](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/chain/tuning/run_tdnn_1d.sh), 17-layer TDNN-F + iVectors|
| 2.8%  | 9.3%   | [RWTH ASR Systems for LibriSpeech: Hybrid vs Attention](https://arxiv.org/abs/1905.03072) | September 2019, Interspeech | encoder-attention-decoder + BPE + Transformer LM (no data augmentation) |
| 3.26% | 10.47% | [Fully Convolutional Speech Recognition](https://arxiv.org/abs/1812.06864) | December 2018 | End-to-end CNN on the waveform + conv LM|
| 3.82% | 12.76% | [Improved training of end-to-end attention models for speech recognition](https://www-i6.informatik.rwth-aachen.de/publications/download/1068/Zeyer--2018.pdf) | Interspeech, Sept 2018 | encoder-attention-decoder end-to-end model |
| 4.28% | | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-TDNN trained with MMI + data augmentation (speed) + iVectors + 3 regularizations |
| 4.83% | | [A time delay neural network architecture for efficient modeling of long temporal contexts](http://speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) | 2015 | HMM-TDNN + iVectors |
| 5.15% | 12.73% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://proceedings.mlr.press/v48/amodei16.pdf) | December 2015 | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 100M parameters trained on 11940h |
| 5.51% | 13.97% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | 2015 | HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |
| 4.8%  | 14.5% | [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/abs/1712.09444) | December 2017 | (Gated) ConvNet for AM going to letters + 4-gram LM |
| 8.01% | 22.49% | same, [Kaldi](http://kaldi-asr.org/) | 2015 | HMM-(SAT)GMM |
| | 12.51% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | 2015 | TDNN + pNorm + speed up/down speech |

### WSJ

(Possibly trained on more data than WSJ.)

| WER eval'92    | WER eval'93    | Paper          | Published | Notes   |
| :------------- | :------------- | :------------- | :-------- | :-----: |
| 5.03% | 8.08% | *Humans* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | December 2015 | *Humans* |
| 2.9% | | [End-to-end Speech Recognition Using Lattice-Free MMI](https://pdfs.semanticscholar.org/dcae/b29ad3307e2bdab2218416c81cb0c4e548b2.pdf) | September 2018 | HMM-DNN LF-MMI trained (biphone) |
| 3.10% | | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://proceedings.mlr.press/v48/amodei16.pdf) | December 2015 | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 100M parameters |
| 3.47% | | [Deep Recurrent Neural Networks for Acoustic Modelling](http://arxiv.org/pdf/1504.01482v1.pdf) | April 2015 | TC-DNN-BLSTM-DNN |
| 3.5%  | 6.8%  | [Fully Convolutional Speech Recognition](https://arxiv.org/abs/1812.06864) | December 2018 | End-to-end CNN on the waveform + conv LM|
| 3.63% | 5.66% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | 2015 | test-set on open vocabulary (i.e. harder), model = HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) |
| 4.1% | | [End-to-end Speech Recognition Using Lattice-Free MMI](https://pdfs.semanticscholar.org/dcae/b29ad3307e2bdab2218416c81cb0c4e548b2.pdf) | September 2018 | HMM-DNN E2E LF-MMI trained (word n-gram) |
| 5.6% | | [Convolutional Neural Networks-based Continuous Speech Recognition using Raw Speech Signal](http://infoscience.epfl.ch/record/203464/files/Palaz_Idiap-RR-18-2014.pdf) | 2014 | CNN over RAW speech (wav) |
| 5.7%  | 8.7%  | [End-to-end Speech Recognition from the Raw Waveform](https://arxiv.org/abs/1806.07098) | June 2018 | End-to-end CNN on the waveform|

### Hub5'00 Evaluation (Switchboard / CallHome)

(Possibly trained on more data than SWB, but test set = full Hub5'00.)

| WER (SWB) | WER (CH) | Paper          | Published | Notes   |
| :-------  | :------- | :------------- | :-------- | :-----: |
| 4.9% | 9.5%  | [An investigation of phone-based subword units for end-to-end speech recognition](https://arxiv.org/abs/2004.04290) | April 2020 | 2 CNN + 24 layers Transformer encoder and 12 layers Transformer decoder model with char BPE and phoneme BPE units. |
| 5.0% | 9.1%  | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | December 2017 | 2 Dense LSTMs + 3 CNN-bLSTMs across 3 phonesets from [previous Capio paper](https://pdfs.semanticscholar.org/d0ec/cd60d800308cd6e59810769b92b40961c09a.pdf) & AM adaptation using parameter averaging (5.6% SWB / 10.5% CH single systems) |
| 5.1% | 9.9%  | [Language Modeling with Highway LSTM](https://arxiv.org/abs/1709.06436) | September 2017 | HW-LSTM LM trained with Switchboard+Fisher+Gigaword+Broadcast News+Conversations, AM from [previous IBM paper](https://arxiv.org/abs/1703.02136)|
| 5.1% |       | [The Microsoft 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1708.06073) | August 2017 | ~2016 system + character-based dialog session aware (turns of speech) LSTM LM |
| 5.3% | 10.1% | [Deep Learning-based Telephony Speech Recognition in the Wild](https://pdfs.semanticscholar.org/d0ec/cd60d800308cd6e59810769b92b40961c09a.pdf) | August 2017 | Ensemble of 3 CNN-bLSTM (5.7% SWB / 11.3% CH single systems)
| 5.5% | 10.3% | [English Conversational Telephone Speech Recognition by Humans and Machines](https://arxiv.org/abs/1703.02136) | March 2017 | ResNet + BiLSTMs acoustic model, with 40d FMLLR + i-Vector inputs, trained on SWB+Fisher+CH, n-gram + model-M + LSTM + Strided (à trous) convs-based LM trained on Switchboard+Fisher+Gigaword+Broadcast |
| 6.3% | 11.9% | [The Microsoft 2016 Conversational Speech Recognition System](http://arxiv.org/pdf/1609.03528v1.pdf) | September 2016 | VGG/Resnet/LACE/BiLSTM acoustic model trained on SWB+Fisher+CH, N-gram + RNNLM language model trained on Switchboard+Fisher+Gigaword+Broadcast |
| 6.3% | 13.3%  | [An investigation of phone-based subword units for end-to-end speech recognition](https://arxiv.org/abs/2004.04290) | April 2020 | 2 CNN + 24 layers Transformer encoder and 12 layers Transformer decoder model with char BPE and phoneme BPE units. Trained only on SWBD 300 hours. |
| 6.6% | 12.2% | [The IBM 2016 English Conversational Telephone Speech Recognition System](http://arxiv.org/pdf/1604.08242v2.pdf) | June 2016 | RNN + VGG + LSTM acoustic model trained on SWB+Fisher+CH, N-gram + "model M" + NNLM language model |
| 6.8% | 14.1% | [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) | April 2019 | Listen Attend Spell |
| 8.5% | 13% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-BLSTM trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + Fisher |
| 9.2% | 13.3% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-TDNN trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + Fisher (10% / 15.1% respectively trained on SWBD only) |
| 12.6% | 16% | [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) | December 2014 | CNN + Bi-RNN + CTC (speech to letters), 25.9% WER if trained _only_ on SWB |
| 11% | 17.1% | [A time delay neural network architecture for efficient modeling of long temporal contexts](http://speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) | 2015 | HMM-TDNN + iVectors |
| 12.6% | 18.4% | [Sequence-discriminative training of deep neural networks](http://www.danielpovey.com/files/2013_interspeech_dnn.pdf) | 2013 | HMM-DNN +sMBR |
| 12.9% | 19.3% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | 2015 | HMM-TDNN + pNorm + speed up/down speech |
| 15% | 19.1% | [Building DNN Acoustic Models for Large Vocabulary Speech Recognition](http://arxiv.org/abs/1406.7806v2) | June 2014  | DNN + Dropout |
| 10.4% | | [Joint Training of Convolutional and Non-Convolutional Neural Networks](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p5609-soltau.pdf) | 2014 | CNN on MFSC/fbanks + 1 non-conv layer for FMLLR/I-Vectors concatenated in a DNN |
| 11.5% | | [Deep Convolutional Neural Networks for LVCSR](http://www.cs.toronto.edu/~asamir/papers/icassp13_cnn.pdf) | 2013 | CNN |
| 12.2% | | [Very Deep Multilingual Convolutional Neural Networks for LVCSR](http://arxiv.org/pdf/1509.08967v1.pdf) | September 2015 | Deep CNN (10 conv, 4 FC layers), multi-scale feature maps |
| 11.8% | 25.7% | [Improved training of end-to-end attention models for speech recognition](https://www-i6.informatik.rwth-aachen.de/publications/download/1068/Zeyer--2018.pdf) | Interspeech, Sept 2018 | encoder-attention-decoder end-to-end model, trained on 300h SWB |

### Rich Transcriptions
| WER RT-02 | WER RT-03 | WER RT-04 | Paper          | Published | Notes   |
| :-------  | :-------- | :-------- | :------------- | :-------- | :-----: |
| 8.1% | 8.0%  |       | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | April 2018 | 2 Dense LSTMs + 3 CNN-bLSTMs across 3 phonesets from [previous Capio paper](https://pdfs.semanticscholar.org/d0ec/cd60d800308cd6e59810769b92b40961c09a.pdf) & AM adaptation using parameter averaging  |
| 8.2% | 8.1%  | 7.7%  | [Language Modeling with Highway LSTM](https://arxiv.org/abs/1709.06436) | September 2017 | HW-LSTM LM trained with Switchboard+Fisher+Gigaword+Broadcast News+Conversations, AM from [previous IBM paper](https://arxiv.org/abs/1703.02136)|
| 8.3% | 8.0%  | 7.7%  | [English Conversational Telephone Speech Recognition by Humans and Machines](https://arxiv.org/abs/1703.02136) | March 2017 | ResNet + BiLSTMs acoustic model, with 40d FMLLR + i-Vector inputs, trained on SWB+Fisher+CH, n-gram + model-M + LSTM + Strided (à trous) convs-based LM trained on Switchboard+Fisher+Gigaword+Broadcast |

### Fisher (RT03S FSH)
| WER     | Paper  | Published | Notes   |
| :------ | :----- | :-------- | :-----: |
| 9.6% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-*BLSTM* trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + SWBD |
| 9.8% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-*TDNN* trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + SWBD |

### TED-LIUM
| WER Test | Paper          | Published | Notes   |
| :------- | :------------- | :-------- | :-----: |
| 5.6% | [The RWTH ASR System for TED-LIUM release 2: Improving Hybrid HMM with SpecAugment](https://arxiv.org/abs/2004.00960) | April 2020 | HMM-BLSTM + iVectors + SpecAugment + sMBR + Transformer LM |
| 6.5% | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | April 2018 | TDNN + TDNN-LSTM + CNN-bLSTM + Dense TDNN-LSTM across two kinds of trees |
| 11.2% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-TDNN trained with LF-MMI + data augmentation (speed perturbation) + iVectors + 3 regularizations |
| 15.3% | [TED-LIUM: an Automatic Speech Recognition dedicated corpus](https://pdfs.semanticscholar.org/1e0b/8416b9d2afb9b1ef87557958ef964cb4472b.pdf) | May 2014 | Multi-layer perceptron (MLP) with bottle-neck feature extraction |

### CHiME6 (multiarray noisy speech)

| WER (constrained LM) | WER (unconstrained LM) | Paper | Published | Notes |
| :---- | :---- | :---- | :---- | :---- |
| 31.0% | 30.5% | [The USTC-NELSLIP Systems for CHiME-6 Challenge](https://chimechallenge.github.io/chime2020-workshop/papers/CHiME_2020_paper_du.pdf) | May 2020 | WPE + SSA + GSS + Data Augment (Speed, Volume) + SpecAugment + 8 AMs fusion (2 Single-feature AM + 6 Multi-feature AM)
| 35.1% | 34.5% | [The IOA Systems for CHiME-6 Challenge](https://chimechallenge.github.io/chime2020-workshop/papers/CHiME_2020_paper_chen.pdf) | May 2020 | WPE + multi-stage GSS (2-stage for train, 3-stage for test) + SpecAugment + Data Augment (Noise, Reverberation, Speed) + 3 AMs fusion (CNN-TDNNF / CNN-TDNN-BLSTM + CNN-BLSTM)
| 35.8% | 33.9% | [The STC System for the CHiME-6 Challenge](https://chimechallenge.github.io/chime2020-workshop/papers/CHiME_2020_paper_medennikov.pdf) | May 2020 | WPE + GSS + SpecAugment + 3 AMs fusion (TDNN-F on MFCC / TDNN-F on fbank gtf + stats + SpecAugment / CNN-TDNNF + stats + SpecAugment + multi-stride self-attention + multi-stream self-attention + sMBR) + MBR Decoding |
| 51.3% | 51.3% | [Baseline](https://arxiv.org/pdf/2004.09249.pdf)| May 2020 | WPE + GSS + Data Augment (Noise, Reverberation, Speed) + TDNNF

### CHiME (noisy speech)

| clean | real | sim | Paper | Published | Notes |
| :------ | :----- | :----- | :----- | :----- | :-----: |
| 3.34% | 21.79% | 45.05% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | December 2015 | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 68M parameters |
| 6.30% | 67.94% | 80.27% | [Deep Speech: Scaling up end-to-end speech recognition](http://arxiv.org/abs/1412.5567) | December, 2014 |  CNN + Bi-RNN + CTC (speech to letters) |

TODO

## PER

### TIMIT

(So far, all results trained on TIMIT and tested on the core test set.)

| PER     | Paper  | Published | Notes   |
| :------ | :----- | :-------- | :-----: |
| 13.8%   | [The Pytorch-Kaldi Speech Recognition Toolkit](https://arxiv.org/abs/1811.07453) | February 2019 | MLP+Li-GRU+MLP on MFCC+FBANK+fMLLR |
| 14.9%   | [Light Gated Recurrent Units for Speech Recognition](https://arxiv.org/abs/1803.10225) | March 2018 | Removing the reset gate in GRU, using ReLU activation instead of tanh and batch normalization |
| 16.5%   | [Phone recognition with hierarchical convolutional deep maxout networks](https://link.springer.com/content/pdf/10.1186%2Fs13636-015-0068-3.pdf) | September 2015 | Hierarchical maxout CNN + Dropout |
| 16.5%   | [A Regularization Post Layer: An Additional Way how to Make Deep Neural Networks Robust](https://www.researchgate.net/profile/Jan_Vanek/publication/320038040_A_Regularization_Post_Layer_An_Additional_Way_How_to_Make_Deep_Neural_Networks_Robust/links/59f97fffaca272607e2f804a/A-Regularization-Post-Layer-An-Additional-Way-How-to-Make-Deep-Neural-Networks-Robust.pdf) | 2017 | DBN with last layer regularization |
| 16.7%   | [Combining Time- and Frequency-Domain Convolution in Convolutional Neural Network-Based Phone Recognition](http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2014.pdf) | 2014 | CNN in time and frequency + dropout, 17.6% w/o dropout |
| 16.8%   | [An investigation into instantaneous frequency estimation methods for improved speech recognition features](https://ieeexplore.ieee.org/abstract/document/8308665) | November 2017 | DNN-HMM with MFCC + IFCC features |
| 17.3%   | [Segmental Recurrent Neural Networks for End-to-end Speech Recognition](https://arxiv.org/abs/1603.00223) | March 2016 | RNN-CRF on 24(x3) MFSC |
| 17.6%   | [Attention-Based Models for Speech Recognition](http://arxiv.org/abs/1506.07503) | June 2015 | Bi-RNN + Attention |
| 17.7%   | [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/abs/1303.5778v1) | March 2013 | Bi-LSTM + skip connections w/ RNN transducer (18.4% with CTC only) |
| 18.0%   | [Learning Filterbanks from Raw Speech for Phone Recognition](https://arxiv.org/abs/1711.01161) | October 2017 | Complex ConvNets on raw speech w/ mel-fbanks init |
| 18.8%   | [Wavenet: A Generative Model For Raw Audio](https://arxiv.org/pdf/1609.03499.pdf) | September 2016 | Wavenet architecture with mean pooling layer after residual block + few non-causal conv layers |
| 23%     | [Deep Belief Networks for Phone Recognition](http://www.cs.toronto.edu/~asamir/papers/NIPS09.pdf) | 2009 | (first, modern) HMM-DBN |

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
 * TDNN-F: a factored form of time delay neural networks (TDNN)
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
 * IFCC: Instantaneous frequency cosine coefficients (https://github.com/siplabiith/IFCC-Feature-Extraction)
 * VGG: very deep convolutional neural networks from Visual Graphics Group, VGG is an architecture of 2 {3x3 convolutions} followed by 1 pooling, repeated
