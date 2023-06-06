# ID_fraud_detection
## Introduction  <br />
In this paper, we propose two anomaly detection models for
identity documents verification, comprising contrastive and adversarial learning frameworks.
The first proposed model learns the representations in a con-
trastive learning manner, named as contrastive based fraud de-
tection model (ContFD), and the second model learns the rep-
resentations based on an adversarial setting, which is named as
constrained-adversary based fraud detection model (AdvFD).
Both models work to well classify authentic (real) and forged
(fake) identity documents.

## Contents  <br />
1- Contrastive based fraud detection model (ContFD). <br />
This model employs an encoder-decoder-classifier sub-networks which enable the model to map the input image into a lower-dimension feature vector, and then to reconstruct the output image. The objective of classifier is to well classify the input image into a real or fake image. 
<img
  src="blob/ContFD.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  <br />
2- Constrained-adversary based fraud detection model (AdvFD).<br /> 
This model is similar to the ContFD model, the ony difference is that the classifier network f(.) is replaced by a onstrained-adversarial model A(.).
<img
  src="blob/AdvFD.png"
  alt="Alt text"
  title="Optional title"
  style="display: inline-block; margin: 0 auto; max-width: 50px">
  <br />

## Installation <br />

pip install -r requirements.txt

git clone https://github.com/malghadi/ID_fraud_detection.git

--# install pytorch, torchvision refer to https://pytorch.org/get-started/locally/  
<br />
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html


## Pre-trained Models <br />


## Description  <br />


## Citation <br />


## Acknowledgement <br />


## Contact  <br />
