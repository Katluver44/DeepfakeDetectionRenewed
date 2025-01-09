# Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes

## Overview
This repository contains the code for the paper titled "Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes" published in AAAI 2025.
[Arxiv](https://arxiv.org/abs/2412.12619) 


## Abstract

Recent advancements in text-to-speech and speech conversion technologies have enabled the creation of highly convincing synthetic speech. While these innovations offer numerous practical benefits, they also cause significant security challenges when maliciously misused. Therefore, there is an urgent need to detect these synthetic speech signals. Phoneme features provide a powerful speech representation for deepfake detection. However, previous phoneme-based detection approaches typically focused on specific phonemes, overlooking temporal inconsistencies across the entire phoneme sequence. In this paper, we develop a new mechanism for detecting speech deepfakes by identifying the inconsistencies of phoneme-level speech features. We design an adaptive phoneme pooling technique that extracts sample-specific phoneme-level features from frame-level speech data. By applying this technique to features extracted by pre-trained audio models on previously unseen deepfake datasets, we demonstrate that deepfake samples often exhibit phoneme-level inconsistencies when compared to genuine speech. To further enhance detection accuracy, we propose a deepfake detector that uses a graph attention network to model the temporal dependencies of phoneme-level features. Additionally, we introduce a random phoneme substitution augmentation technique to increase feature diversity during training. Extensive experiments on four benchmark datasets demonstrate the superior performance of our method over existing state-of-the-art detection methods.


## Requirements
- Python 3.8+
- PyTorch 1.10+
- librosa 0.9.0+
- numpy 1.21+
- scipy 1.7+
- tqdm 4.62+

## Usage


One can run the following commands to train or test our method.
```bash
python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2019_LA'  -v 0;\
python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2019_LA'  -t 1 -v 0;\


python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2021_LA'  -v 0;\
python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2021_LA'  -t 1 -v 0;\


python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2021_inner'  -v 0;\
python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2021_inner'  -t 1 -v 0;\
python train.py --gpu 0 --cfg 'OursPhonemeGAT/ASV2021_inner'  -t 1 -v 0 --test_noise 1 --test_noise_level 20 --test_noise_type 'bg';\



python train.py --gpu 0 --cfg 'OursPhonemeGAT/MLAAD_cross_lang'  -v 0;\
python train.py --gpu 0 --cfg 'OursPhonemeGAT/MLAAD_cross_lang'  -t 1 -v 0;\
```