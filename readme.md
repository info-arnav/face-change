# Face Changing Program

This program uses special tools like deepface, insightface, and face recognition to help you choose and change faces in a group picture. It's like swapping faces to create the look you want, and these tools make it super easy!

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
