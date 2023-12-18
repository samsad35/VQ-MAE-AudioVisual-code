
# A vector quantized masked autoencoder for audiovisual speech emotion recognition
[![Generic badge](https://img.shields.io/badge/<STATUS>-<in_progress>-<COLOR>.svg)]()
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://samsad35.github.io/site-mdvae/)

[comment]: <> ([![PyPI version fury.io]&#40;https://badge.fury.io/py/ansicolortags.svg&#41;]&#40;https://test.pypi.org/project/&#41;)


This repository contains the code associated with the following publication:
> **A vector quantized masked autoencoder for audiovisual emotion recognition**<br> Samir Sadok, Simon Leglaive, Renaud Séguier<br>Face and gesture 2024.

If you use this code for your research, please cite the above paper.

Useful links:
- [Abstract](https://arxiv.org/abs/2305.03568)
- [Paper]()
- [Demo website with qualitative results](https://samsad35.github.io/site-mdvae/)

## Setup 
- [ ] Pypi: (Soon) 

[comment]: <> (  - ``````)
- [ ] Install the package locally (for use on your system):  
  - In VQ-MAE-speech directoy: ```pip install -e .```
- [x] Virtual Environment: 
  - ```conda create -n vq_mae_av python=3.8```
  - ```conda activate vq_mae_av```
  - In VQ-MAE-speech directoy: ```pip install -r requirements.txt```

## Usage
* To do:
  * [x] Training VQ-VAE-Speech
  * [x] Training VQ-VAE-Visual
  * [X] Training VQ-MAE-AV
  * [X] Fine-tuning and classification for emotion recognition

### 1) Training Speech VQ-VAE-Specch in unsupervised learning

![VQ-VAE-audio](images/tokens-audio.svg)

See the code [train_speech_vqvae.py](train_speech_vqvae.py)
- You can download our pre-trained speech VQ-VAE [following link]() (released soon).

### 2) Training Speech VQ-VAE-Specch in unsupervised learning

![VQ-VAE-visual](images/tokens-visual.svg)

See the code [train_visual_vqvae.py](train_visual_vqvae.py)
- You can download our pre-trained speech VQ-VAE [following link]() (released soon).

### 2) Training VQ-MAE-Speech in self-supervised learning
![VQ-MAE](images/overview-new.svg)

See the code [train_vq_mae_av.py](train_vq_mae_av.py)
- Pretrained models (released soon)

| Model         	| Encoder depth    	| 
|---------------	|---------------------	|
| VQ-MAE-AV 	| [6]() - [12]() - [16]() - [20]() 	|

### 3) Fine-tuning and classification for emotion recognition task

- __cross-validation | Speaker independent__ Follow the file "[classification_speaker_independent.py](classification_speaker_independent.py)".
- __80%/20% | Speaker dependent__ Follow the file "[classification_speaker_dependent.py](classification_speaker_dependent.py)".

```

## License
GNU Affero General Public License (version 3), see LICENSE.txt.