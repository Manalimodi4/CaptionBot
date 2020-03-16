# CaptionBot-Image contextual analysis using RNN and CNN
## Description
This project aims to identify the purpose behind a visual depiction of an image captured, analyze the context behind a visual image and generate an artistic caption for the same. The resulted caption will not necessarily be descriptive but rather contextual and creative. There doesn't exist a direct mapping between image and its corresponding description generated but an abstract mapping that denotes the image in to sentences which is very much artistic and aiming to exhibit a kind of computational creativity.A neural network is trained using Flickr8k dataset so that this pretrained model can be used to generate caption.A user interface is built for users to upload image or paste a remote image URL or image can be captured directly through real time camera.

### Procedure:
#### 1.Download and Install latest version of Python and set the environment variable
#### 2.Download Anaconda or Google Colab can be used if you don't have a GPU.
#### 3.Download Flickr8k dataset which contains of 8k images and 5 relevant captions for each image 
#### 4.Download the following libraries used in this project
 • tensorflow-1.15.0
 • keras-2.2.5
 • numpy-1.17.5
 • pandas-0.25.3
 • matplotlib-3.1.3
 • flask-2.10.1
#### 5. For obtaining vector representations for words download GloVe.6B.200d
#### 6. Download the code from the github repository: https://github.com/Manalimodi4/CaptionBot
#### 7. The code for training the model is contained in Trained_model.ipynb
#### 8. The code for flask web service is in server.py
#### 9. UI for uploading files is contained in index.html
#### 10. Run server.py and the curl command in terminal
#### 11. Open browser and type localhost:5000



