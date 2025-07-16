# SnapMoGen: Human Motion Generation from Expressive Texts

<p align="left">
  <a href='https://www.arxiv.org/abs/2507.09122'>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href='https://snap-research.github.io/SnapMoGen/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
  <a href='https://huggingface.co/datasets/Ericguo5513/SnapMoGen'> 
    <img src='https://img.shields.io/badge/Dataset-SnapMoGen-blue'></a>
</p>

![teaser_image](https://github.com/snap-research/SnapMoGen/blob/gh_pages/static/images/result.png)

If you find our code or paper helpful, please consider **starring** this repository and citing the following:

```
@misc{snapmogen2025,
      title={SnapMoGen: Human Motion Generation from Expressive Texts}, 
      author={Chuan Guo and Inwoo Hwang and Jian Wang and Bing Zhou},
      year={2025},
      eprint={2507.09122},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.09122}, 
}
```

## :postbox: News

📢 **2023-11-29** --- Initialized the webpage and git project.

## :round_pushpin: Getting Started

  
### 1.1 Set Up Conda Environment
  
```sh
conda env create -f environment.yml
conda activate momask-plus
```

#### 🔁 Alternative: Pip Installation
If you encounter issues with Conda, you can install the dependencies using pip:

```sh
pip install -r requirements.txt
```

✅ Tested on Python 3.8.20.

### 1.2 Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
> (For evaluation only.)
```
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1qW_VVDbFy9E05U2E_N95zi-tDWrF77zw?usp=drive_link) to download the models and evaluators mannually.

### 1.3 Download the Datasets

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the dataset to your data folder:

```
cp -r ./HumanML3D/ your_data_folder/HumanML3D
```

**SnapMoGen** - Download the data from [huggingface](https://huggingface.co/datasets/Ericguo5513/SnapMoGen), then place it in the following directory:

```
cp -r ./SnapMoGen your_data_folder/SnapMoGen
```

## :rocket: Play with MoMask++

> Remember to update the ``data.root_dir`` in all the ``config/*.yaml`` files -  with your own data directory path.

### 2.1 Motion Generation 

To generate motion from your own text prompts, use:

```
python gen_momask_plus.py
```
You can modify the inference configuration (e.g., number of diffusion steps, guidance scale, etc.) in ``config/eval_momaskplus.yaml``.

### 2.2 Evaluation

Run the following scripts for quantitive evaluation:

```sh
python eval_momask_plus_hml.py    # Evaluate on HumanML3D dataset
python eval_momask_plus.py        # Evaluate on SnapMoGen dataset
```

### 2.3 Training

There are two main components in MoMask++, a multi-scale residual motion VQVAE and a generative masked Transformer.

> All checkpoints will be stored under ``/checkpoint_dir``.

#### Multi-scale Motion RVQVAE

```sh
python train_rvq_hml.py           # Train RVQVAE on HumanML3D
python train_rvq.py               # Train RVQVAE on SnapMoGen
```

Configuration files:
* ``config/residual_vqvae_hml.yaml`` (for HumanML3D)
* ``config/residual_vqvae.yaml`` (for SnapMoGen)

#### Generative Masked Transformer


```sh
python train_momask_plus_hml.py   # Train on HumanML3D
python train_momask_plus.py       # Train on SnapMoGen
```

Configuration files:
* ``config/train_momaskplus_hml.yaml`` (for HumanML3D)
* ``config/train_momaskplus.yaml`` (for SnapMoGen)
  
> Remember to change ``vq_name`` and ``vq_ckpt`` to your VQ name and VQ checkpoint in these two configuration files.
> Training accuracy at around 0.25 is normal.

  
#### Global Motion Refinement

We use a separate lightweight root motion regressor to refine the root trajectory. In particular, this regressor is trained given local motion features to predict root linear velocities. During motion generation, we use this regressor to re-predict the resulting root trajectories which effectively reduces sliding feet.

## :clapper: Visualization

All animations were manually rendered in **Blender** using **Bitmoji** characters.  
An example character is available [here](https://drive.google.com/file/d/1tRZHp0jXdvB3n7LDQPccM1KzwygOJF1x/view?usp=drive_link), and we use [this Blender scene](https://drive.google.com/file/d/16SbrnG9JsJ2w7UwCFmh10PcBdl6HxlrA/view?usp=drive_link) for animation rendering.

---

### Retargeting

We recommend using the [Rokoko Blender add-on](https://www.rokoko.com/integrations/blender) (v1.4.1) for seamless motion retargeting.

> ⚠️ Note: All motions in **SnapMoGen** use **T-Pose** as the rest pose.

If your character rig is in **A-Pose**, use the ``rest_pose_retarget.py`` to convert between T-Pose and A-Pose rest poses:


## Acknowlegements

We sincerely thank the open-sourcing of these works where our code is based on: 

[MoMask](https://github.com/EricGuo5513/momask-codes), [VAR](https://github.com/FoundationVision/VAR), [deep-motion-editing](https://github.com/DeepMotionEditing/deep-motion-editing), [Muse](https://github.com/lucidrains/muse-maskgit-pytorch), [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MDM](https://github.com/GuyTevet/motion-diffusion-model/tree/main) and [MLD](https://github.com/ChenFengYe/motion-latent-diffusion/tree/main)

### Misc
Contact guochuan5513@gmail.com for further questions.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=snap-research/SnapMoGen&type=Date)](https://www.star-history.com/#snap-research/SnapMoGen&Date)
