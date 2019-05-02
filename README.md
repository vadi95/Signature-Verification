# Signature-Verification

Give a set of forged and genuine signature images determines if a signature is genuine or forged.
<br><br>
## Features Used ##
* SIFT<br>
* Convex Hull area / Bounding Rectangle area<br>
* Aspect ratio of bounding rectangle<br>
* Countour area / Bounding Rectangle area<br>
* Perceptual hash of the image

## Model used ##

Linear SVM <br>
Precision ~ 0.92<br>
Recall - 1.0

## Run ##

#### Docker

Install docker - https://docs.docker.com/install/ <br>
`docker build -t signature-verification .` <br>
`docker run signature-verification` <br>

#### Python virtual env (python 2.7)

`pip install -r requirements.txt`  <br>
`python run.py`
