## Introduction

This repository contains our implementation of the paper published in IEEE Internet of Things Journal, "__MultiSiFer: Detecting Multiple-Speaker Fake Voice Without Speaker-Irrelative Features__". 
[paper link here](https://ieeexplore.ieee.org/document/11316509)

### Description

MultiSiFer, a fake voice detector that leverages a pre-trained speech representation model to enhance human voice feature learning. To improve its multi-speaker detection ability, we build the first multi-speaker voice dataset, **multi_speaker_LA**, for fake voice detection and fine-tune MultiSiFer using this new dataset. The dataset is built based on the ASVspoof2019 LA dataset and contains 24,000 multi-speaker voice samples.

[multi_speaker_LA download here](https://drive.google.com/file/d/16nLuxIYemwNcov2KuX0eHHwmthYOa55z/view?usp=sharing)
### Framework

The framework of our approach is shown in the figure below.

![Framework](Framework.png "Framework")

The framework consists of **upstream** and **downstream** networks:

1. **Upstream Network**:  
   - Utilizes a **pre-trained self-supervised speech representation model** as the feature extractor to enhance learning of human voice feature distribution.  
   - Introduces a **layer selection module** to filter specific feature information and applies a **feature fusion operation** to aggregate information from multiple feature maps.

2. **Downstream Network**:  
   - Employs a **CNN-based architecture** to further extract and aggregate feature information, emphasizing key features while suppressing irrelevant noise.  
   - Incorporates a **BiLSTM module** to capture frame-level feature representations, followed by a linear layer to map the BiLSTM output to the final result.

## Installation

MultiSiFer is a fake voice detector that can be installed using the following steps:

First, download the MultiSiFer source code from GitHub.

    $ git clone https://github.com/0Rambler0/MultiSiFer.git


Create a conda environment and activate it, and then install the dependencies that will be used.

    $ conda create -n multisifer python=3.8
    $ conda activate multisifer  
    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
    $ conda install -c conda-forge tensorboard

Then, navigate to the MultiSiFer folder and run the following command to install all the dependencies that are provided in requirement.txt:    

    $ pip install -r requirements.txt

__requirement.txt__:
```
torchcontrib
soundfile
librosa
s3prl
peft
pydub
```

## Dataset

We use ASVspoof 2019 dataset, 2021 LA and DF evaluation database in our experiment, which can can be downloaded from:

[19LA download here](https://datashare.is.ed.ac.uk/handle/10283/3336)

[21LA download here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

[21DF download here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

In our experiment, we used these data after silence processing. To get the same process with us, please change the path in the corresponding file and run the command to get a new dataset called **LA_silence**:

```
$ python3 dataset_utils/silence_sox.py
$ cp -r LA/ASVspoof2019_LA_cm_protocols LA_silence
```

To build the **multi_speaker_LA** dataset, we use the corresponding utils to  filter out samples whose durations are less than two seconds and create the mix audio and put these files together:

```
$ python3 dataset_utils/get_long_audio.py
$ python3 dataset_utils/mix_all_true.py
$ python3 dataset_utils/mixture_spoof_true.py
$ python3 dataset_utils/move_audio.py
```

You can directly download from the link in the [Description](#description) to use them.


## Config file

__Configuration option__
*   "__database_path__" should be the corresponding dataset path;
*   "__asv_score_path__" means path where the ASV system score file is stored;
*   "__model_path__" means path where the model weight is stored;
*   "__batch_size__" means number of data samples processed in each iteration (one forward and back propagation);
*   An epoch in machine learning refers to one complete pass through the entire training dataset. The parameter "__num_epochs__" means the number of epochs;
*   "__loss__" is the loss function of the optimization;
*   "__track__" should be LA or DF;
*   "__eval_all_best__" determines whether or not a complete evaluation is immediately performed on the evaluation set when a new optimal model is discovered;
*   "__cudnn_deterministic_toggle__" used to control the determinism of model training and inference;
*   "__cudnn_benchmark_toggle__" used to control whether CuDNN enables automatic algorithm optimization;

_model_config:_

*   "__architecture__" represents the architecture or type of model;
*   "__is_lora__" specifies the layer in the model where the LoRA module should be injected;
*   "__inject_modules__" specifiees the layer in the model where the LoRA module should be injected;
*   "__lora_layers__" is a list specifying the layers in the model where LoRA modules need to be injected;
*   "__layers__" is a list representing the configuration of the layers to be used in the model.

_optim_config:_

*   "__optimizer__" specifies the type of optimizer;
*   "__lr__" means learning rate;
*   "__weight_decay__", weight decay is a technique to prevent model overfitting.

We provide 5 config files for 5 conditions, include training the original model (**train_model.conf**), fine tuning the original model (**fine_tuning.conf**), evaluate the fine tuning model on multi_speaker_LA dataset (**eval_model.conf**), evaluate the fine tuning model on 19LA dataset after silence process (**basic19_eval_model.conf**), and evaluate the fine tuning model on 2021 LA DF dataset after silence process (**fine_tuning_basic_21.conf**). 

If you want to directly use the config file we provide, you simply need to change the relative path in config file before your experiments. And you can also design your own config file to cater to your needs.

## Training original model

To train the model, please run:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config/train_model.conf --sr_model xls_r_300m --comment train

We also provide a pre-trained model, named ***best.pth***, which can be downloaded from [here](https://drive.google.com/file/d/17rmaa-_3JwjoxG76Wc3ALzuYzar8a7GC/view?usp=drive_link), and place it in the ***weight*** folder before use.

## Fine tuning

To **fine tuning** the original model, please run:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config/fine_tuning.conf --sr_model xls_r_300m --comment fine_tuning --fine_tuning

We also provide a weight after doing fine tuning, named ***fine_tuning_best.pth***, which can be downloaded from [here](https://drive.google.com/file/d/1pTrcfKqQkprytX3SRYEJyZVuurVSawx2/view?usp=drive_link), and place it in the ***weight*** folder before use.

To **evaluate** the fine tuning model on the **multi_speaker_LA** dataset, please run:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config/eval_model.conf --sr_model xls_r_300m --comment fine_tuning_eval --eval

To **evaluate** the fine tuning model on the **LA_silence** dataset, please run:

    $ CUDA_VISIBLE_DEVICES=0 python3 main.py --config ./config/basic19_eval_model.conf --sr_model xls_r_300m --comment fine_tuning_eval --basic19_eval

## Evaluation on 2021 LA DF

To **evaluate** the fine tuning model on the **ASVspoof2021_DF_eval** dataset, please run to create the score file:

    $ CUDA_VISIBLE_DEVICES=0 python eval_2021.py  --config ./config/fine_tuning_basic_21.conf --sr_model xls_r_300m --comment eval_21DF --eval_model_weights ./weights/fine_tuning_best.pth --track DF --eval

To **evaluate** the fine tuning model on the **ASVspoof2021_LA_eval** dataset, please run to create the score file:

    $ CUDA_VISIBLE_DEVICES=0 python eval_2021.py  --config ./config/fine_tuning_basic_21.conf --sr_model xls_r_300m --comment eval_21LA --eval_model_weights ./weights/fine_tuning_best.pth --track LA --eval

To download ASVspoof 2021 dataset keys (labels) and metadata, please run:

    $ bash eval_package_2021/download.sh

To use the following utils, you also need to install the packages below:

    $ pip install pandas
    $ pip install matplotlib
    


Then, change the **cm-score-file** and **track** and use the downloaded keys to calculate EER (For 21LA, also calculate min t-DCF)

    $ python eval_package_2021/main.py --cm-score-file score-file-path --subset eval --track DF or LA


## License
```
MIT License

Copyright (c) 2026 0Rambler0

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Our Previous Work

__2025__

SiFMimicEvader: Evading Fake Voice Detection with Adversarial Neural Mimicry Attacks (MM '25) [<u>paper link here</u>](https://dl.acm.org/doi/10.1145/3746027.3755595) 

__2024__

What's the Real: A Novel Design Philosophy for Robust AI-Synthesized Voice Detection (MM '24) [<u>paper link here</u>](https://dl.acm.org/doi/10.1145/3664647.3681100) 
[<u>code here</u>](https://github.com/0Rambler0/SiFSafer)

Ghost-in-Wave: How Speaker-Irrelative Features Interfere DeepFake Voice Detectors (ICME) [<u>paper link here</u>](https://ieeexplore.ieee.org/document/10688273)

__2023__

SiFDetectCracker: An Adversarial Attack Against Fake Voice Detection Based on Speaker-Irrelative Features (MM '23) [<u>paper link here</u>](https://dl.acm.org/doi/10.1145/3581783.3613841) [<u>code here</u>](https://github.com/0Rambler0/SiFDetectCracker)

Hidden-in-Wave: A Novel Idea to Camouflage AI-Synthesized Voices Based on Speaker-Irrelative Features (ISSRE) [<u>paper link here</u>](https://ieeexplore.ieee.org/document/10301243)

## Contact

For any query regarding this repository, please contact:

* Xuan Hai: haix2024@lzu.edu.cn
* Xin Liu: bird@lzu.edu.cn

## Citation

If you use this code in your research please use the following citation:

```
@ARTICLE{11316509,
  author={Liu, Xin and Hai, Xuan and Yu, Ziyao and Zhang, Zihao and Fei, Qingyuan and Zhou, Qinguo},
  journal={IEEE Internet of Things Journal}, 
  title={MultiSiFer: Detecting Multiple-Speaker Fake Voice Without Speaker-Irrelative Features}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Detectors;Feature extraction;Human voice;Robustness;Watermarking;Text to speech;Mel frequency cepstral coefficient;Philosophical considerations;Overfitting;Internet of Things;AI-Synthesized Voice;DeepFake;AI-synthesized voice detection;ASVspoof},
  doi={10.1109/JIOT.2025.3648834}}
```



