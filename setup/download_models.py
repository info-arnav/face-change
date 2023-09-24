import gdown

gdown.download(url="https://drive.google.com/file/d/1TDkhJhQkpnKHJlC0Wf4sqznyEhvLtPPa/view?usp=sharing", output="models/inswapper_128.onnx", quiet=False, fuzzy=True)
gdown.download(url="https://drive.google.com/file/d/1PLR98TnHH_u6vLu2LKto_UwJZJbK5mzq/view?usp=sharing", output="models/GFPGANv1.4.pth", quiet=False, fuzzy=True)