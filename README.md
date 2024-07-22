# Improving 3D human pose estimators with weakly supervised losses and biomechanical pose prior regularizers
<p align="center"><img src="images/semi_supv_hpe.png" width="50%" alt="" /></p>

This is the implementation of the approach described in the papers:
> Lawrence Amadi and Gady Agam. [PosturePose: Optimized Posture Analysis for Semi-Supervised Monocular 3D Human Pose Estimation](https://www.mdpi.com/1424-8220/23/24/9749). Intelligent Sensors Special Issue on “Deep Learning Applications for Pose Estimation and Human Action Recognition”, Sensors, 2023.

> Lawrence Amadi and Gady Agam. [Boosting the Performance of Weakly-Supervised 3D Human Pose Estimators with Pose Prior Regularizers](https://ieeexplore-ieee-org.ezproxy.gl.iit.edu/document/9897790). The 29th IEEE International Conference on Image Processing (ICIP, 2022).

We use a combination weakly supervised losses and biomechanical pose prior regularizers to better optimize VideoPose3D estimator in a limited annotated training data scenario (with significantly more unlabelled pose data than labelled data). Therefore, this repo extends the source code of VideoPose3D available at https://github.com/facebookresearch/VideoPose3D

### Results on Human3.6M
Training VideoPose3D with additional weakly supervised losses and regularizers. Evaluated under Protocol 1 (mean per-joint position error), Protocol 2 (mean-per-joint position error after rigid alignment) and J-MPBOE (Joint-propagated mean-per-bone orientation error). The table shows the outcome of training with full supervision on S1 pose data (16% of Human3.6M training data) and weak supervision on S5-8.

| WS. Losses | 2D Detections | Receptive Field | Error (P1) | Error (P1) | J-MPBOE |
|:-------|:-------:|:-------:|:-------:|:-------:|:-------:|
| BSE + BPC + JMC | GT | 1 frame | 52.6 mm | 37.3 mm | 40.6 mm |
| BSE + BPC + JMC + MvP&P | HRNet | 1 frame | 54.0 mm | 41.5 mm | 49.1 mm |
| BSE + BPC + JMC + MvP&P | GT | 1 frame | 43.5 mm | 32.7 mm | 37.2 mm |
| BSE + BPC + JMC | GT | 27 frames | 50.1 mm | 36.8 mm | 40.3 mm |
| BSE + BPC + JMC + MvP&P | GT | 27 frames | 52.4 mm | 39.7 mm | 47.5 mm |
| BSE + BPC + JMC + MvP&P | HRNet | 27 frames | 42.2 mm | 31.8 mm | 36.7 mm |

#### Illustration of Bone Orientation Alignment necessary for JMC and MPBOE
<p align="center"><img src="images/rboa_viz_16.png" width="70%" alt="" /></p>

#### Visual comparison of Protocol 2 (P-MPJPE) and bone-orientation alignment-based errors (MPBOE and J-MPBOE)
<p align="center"><img src="images/posture_protocols.png" width="70%" alt="" /></p>

## Quick start
To get started, follow the instructions in https://github.com/facebookresearch/VideoPose3D to set up VideoPose3D (and access Human3.6M dataset). Then clone this repository and copy over files to the VideoPose3D source code; replacing duplicate files with our version.

### Dependencies
Make sure you have the following dependencies installed before proceeding:
- Python 3+ distribution
- PyTorch >= 0.11.0

### Generating Biomechanical Pose Priors
The pretrained models can be downloaded from AWS. Put `pretrained_h36m_cpn.bin` (for Human3.6M) and/or `pretrained_humaneva15_detectron.bin` (for HumanEva) in the `checkpoint/` directory (create it if it does not exist).
```sh
mkdir checkpoint
cd checkpoint
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_humaneva15_detectron.bin
cd ..
```
[//]: # (This is a comment)
#### Bone Proportion Constraint (BPC) biomechanical regularizer derived from S1 pose data
<p align="center"><img src="images/s1_perprp_likli.png" width="70%" alt="" /></p>

#### Joint Mobility Constraint (JMC) biomechanical regularizer derived from S1 pose data
<p align="center"><img src="images/s1_perjnt_logli.gif" width="70%" alt="" /></p>

![](images/demo_temporal.gif)

### Semi-supervised training from scratch
If you want to reproduce the results of our pretrained models, run the following commands.

For Human3.6M:
```
python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
```
By default the application runs in training mode. This will train a new model for 80 epochs, using fine-tuned CPN detections. Expect a training time of 24 hours on a high-end Pascal GPU. If you feel that this is too much, or your GPU is not powerful enough, you can train a model with a smaller receptive field, e.g.
- `-arc 3,3,3,3` (81 frames) should require 11 hours and achieve 47.7 mm. 
- `-arc 3,3,3` (27 frames) should require 6 hours and achieve 48.8 mm.

You could also lower the number of epochs from 80 to 60 with a negligible impact on the result.


### Semi-supervised training without camera parameters
To perform semi-supervised training, you just need to add the `--subjects-unlabeled` argument. In the example below, we use ground-truth 2D poses as input, and train supervised on just 10% of Subject 1 (specified by `--subset 0.1`). The remaining subjects are treated as unlabeled data and are used for semi-supervision.
```
python run.py -k gt --subjects-train S1 --subset 0.1 --subjects-unlabeled S5,S6,S7,S8 -e 200 -lrd 0.98 -arc 3,3,3 --warmup 5 -b 64
```
This should give you an error around 65.2 mm. By contrast, if we only train supervised



## License
This work is licensed under CC BY-NC. Third-party source code and datasets are subject to their respective licenses.
If you use our code/models in your research, please cite our paper:
```
@inproceedings{amadi:posturepsoe:2023,
  title={PosturePose: Optimized Posture Analysis for Semi-Supervised Monocular 3D Human Pose Estimation},
  author={Amadi, Lawrence and Agam, Gady},
  journal={Sensors},
  year={2023}
}
```
```
@inproceedings{amadi:posereg:2022,
  title={Boosting the Performance of Weakly-Supervised 3d Human Pose Estimators With Pose Prior Regularizers},
  author={Amadi, Lawrence and Agam, Gady},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  year={2022}
}
```