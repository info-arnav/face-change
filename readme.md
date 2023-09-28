# Face Changing Program

This program uses special tools like deepface, insightface, and face recognition to help you choose and change faces in a group picture.

# Execution

Repo Setup

```
conda create --name Facechange python=3.7
conda activate Facechange
pip install -r requirements.txt
mkdir data frames middleware models output static temp text text-box
python setup/download_models.py
wget https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata
gunzip eng.traineddata.gz
sudo mv -v eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/
```

Execution:

```
conda activate Facechange
python demo01.py
```
