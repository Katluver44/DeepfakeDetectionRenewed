# Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes

## Overview
This repository contains the code for the paper titled "Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes" published in AAAI 2025.
[Arxiv](https://arxiv.org/abs/2412.12619) 


## Abstract

Recent advancements in text-to-speech and speech conversion technologies have enabled the creation of highly convincing synthetic speech. While these innovations offer numerous practical benefits, they also cause significant security challenges when maliciously misused. Therefore, there is an urgent need to detect these synthetic speech signals. Phoneme features provide a powerful speech representation for deepfake detection. However, previous phoneme-based detection approaches typically focused on specific phonemes, overlooking temporal inconsistencies across the entire phoneme sequence. In this paper, we develop a new mechanism for detecting speech deepfakes by identifying the inconsistencies of phoneme-level speech features. We design an adaptive phoneme pooling technique that extracts sample-specific phoneme-level features from frame-level speech data. By applying this technique to features extracted by pre-trained audio models on previously unseen deepfake datasets, we demonstrate that deepfake samples often exhibit phoneme-level inconsistencies when compared to genuine speech. To further enhance detection accuracy, we propose a deepfake detector that uses a graph attention network to model the temporal dependencies of phoneme-level features. Additionally, we introduce a random phoneme substitution augmentation technique to increase feature diversity during training. Extensive experiments on four benchmark datasets demonstrate the superior performance of our method over existing state-of-the-art detection methods.


## Requirements

```bash
pip install -r requirements.txt
```
Actually, the package versions are not strict. Maybe the latest versions of torch and pytorch_lightning can still work.


You have to install the required Python packages using pip:
```bash
pip install torch torchaudio torchvision librosa einops transformers pytorch_lightning  phonemizer
pip install torch-yin
```
where, you can use the python version 3.9 or higher (My tests are using python 3.9).

Besides, you need to install the `ffmpeg` in your system. Note, Torchaudio may need the version of `ffmpeg` to be lower than 7, and install ffmpeg first and then install torchaudio.



## Usage

Please see the `demo.ipynb` for usage details. You can download the pretrained phoneme recognition model in [google drive](https://drive.google.com/file/d/1SbqynkUQxxlhazklZz9OgcVK7Fl2aT-z/view?usp=drive_link).



## Acknowledgments

Please feel free to contact me (zkyhitsz@gmail.com) if you have any questions. 


Please cite the following paper if you use this code:
```bibtex
@article{Zhang_Hua_Lan_Zhang_Guo_2025, 
  title={Phoneme-Level Feature Discrepancies: A Key to Detecting Sophisticated Speech Deepfakes}, 
  volume={39}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/32093}, 
  DOI={10.1609/aaai.v39i1.32093}, 
  number={1}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Zhang, Kuiyuan and Hua, Zhongyun and Lan, Rushi and Zhang, Yushu and Guo, Yifang}, 
  year={2025}, 
  month={Apr.}, 
  pages={1066-1074} 
}
```