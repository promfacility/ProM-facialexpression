# ProM-facialexpression
It recognizes the facial expression of each face in the image. It works on the onboard camera of a Nvidia Jetson TX2

The code is inspired by [this](https://github.com/serengil/tensorflow-101) repo


## Instructions
1. Download ```haarcascade_frontalface_default.xml``` from [here](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in ```model/```
2. Download structure and weights from [here](https://github.com/serengil/tensorflow-101/tree/master/model)
3. Run ```python3 src/main.py```