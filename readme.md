# Face Changing Program

This program uses special tools like deepface, insightface, and face recognition to help you choose and change faces in a group picture.

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
