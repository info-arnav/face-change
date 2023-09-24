# Face Changing Program

This program helps you choose the faces you want to change in a group image and change them with the faces you want to with a pretrained model using the library insightface.

# Execution

Repo Setup

```
conda create --name Facechange python=3.7
conda activate Facechange
pip install -r requirements.txt
mkdir data frames middleware models output static temp
python setup/download_models.py
```

Execution:

```
conda activate Facechange
python demo01.py
```
