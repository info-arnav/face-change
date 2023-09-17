# Face Changing Program

This program helps you choose the faces you want to change in a group image and change them with the faces you want to with a pretrained model using the library insightface.

# Repo Setup

Create 2 folders in home directory:
middleware
frames

Install : https://drive.google.com/file/d/1TDkhJhQkpnKHJlC0Wf4sqznyEhvLtPPa/view?usp=sharing

Save the above file in the repo directory with the name inswapper_128.onnx

Install : https://drive.google.com/file/d/1PLR98TnHH_u6vLu2LKto_UwJZJbK5mzq/view?usp=sharing

Save the above file in the repo directory with the name GFPGANv1.4.pth

# Execution

If you run the following commands, you can skip the repo setup process

```
conda create --name Facechange python=3.7
conda activate Facechange
pip install -r requirements.txt
mkdir frames
mkdir middleware
gdown https://drive.google.com/file/d/1TDkhJhQkpnKHJlC0Wf4sqznyEhvLtPPa/view?usp=sharing
gdown https://drive.google.com/file/d/1PLR98TnHH_u6vLu2LKto_UwJZJbK5mzq/view?usp=sharing
python demo01.py
```

The images in the demo.py file can be changed. The first parameter is the group picutre and the second parameter is array of faces that you want to use in order to replace.
