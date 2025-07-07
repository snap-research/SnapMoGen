# OmniMotion: Human Motion Generation from Expressive Texts

<p align="left">
  <a href=''>
    <img src='https://img.shields.io/badge/Arxiv-Pdf-A42C25?style=flat&logo=arXiv&logoColor=white'></a>
  <a href=''>
    <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a>
  <a href=''>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=white'></a>
</p>

![teaser_image](./static/images/result.png)

If you find our code or paper helpful, please consider starring our repository and citing:
```
xxx
```

## :postbox: News

📢 **2023-11-29** --- Initialized the webpage and git project.

## :round_pushpin: Get You Ready

  
### 1. Conda Environment
  
```sh
conda env create -f environment.yml
conda activate momask-plus
```

#### Alternative: 
In case you have trouble installing by Conda, you can still install through pip.

```sh
pip install -r requirements.txt
```

We tested this with Python 3.8.20.

### 2. Models and Dependencies

#### Download Pre-trained Models
```
bash prepare/download_models.sh
```

#### Download Evaluation Models and Gloves
For evaluation only.
```
bash prepare/download_evaluator.sh
bash prepare/download_glove.sh
```

#### Troubleshooting
To address the download error related to gdown: "Cannot retrieve the public link of the file. You may need to change the permission to 'Anyone with the link', or have had many accesses". A potential solution is to run `pip install --upgrade --no-cache-dir gdown`, as suggested on https://github.com/wkentaro/gdown/issues/43. This should help resolve the issue.

#### (Optional) Download Manually
Visit [[Google Drive]](https://drive.google.com/drive/folders/1sHajltuE2xgHh91H9pFpMAYAkHaX9o57?usp=drive_link) to download the models and evaluators mannually.

### 3. Get Data

**HumanML3D** - Follow the instruction in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git), then copy the dataset to our repository:

```
cp -r ./HumanML3D/ ./data/humanml3d
```

**OmniMotion** - Download the data from [huggingface](https://huggingface.co/datasets/Ericguo5513/OmniMotion), then copy the dataset to our repository:

```
cp -r ./OmniMotion ./data/omnimotion
```

## :rocket: Play with Pre-trained Model

If you want to generate motions given customized text prompt, try the demos in ``gen_momask_plus.py``:

```
python gen_momask_plus.py
```

Check ``config/eval_momaskplus.yaml`` for inference configration such as ``number of steps`` and ``guidance scale``.

Run the following scripts for evaluation 
