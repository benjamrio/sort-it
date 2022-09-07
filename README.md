# Sort-it

## A machine-learning student project from 2020

Planet matters. **Stream** with your webcam or **upload** an image, and our application will tell you in wich trash you have to throw it.

## Haar-Cascade Manually Trained

We created our own cascade classifier with Cascade Trainer GUI [1] on a dataset of 2527 positives and 6049 negatives. The training took place on a 4 cores i5-7300HQ @2.50GHz cpu chip running at full speed more than 4 hours.
Best scale factor and close neighbours values seem to be around (1.5, 6).
The `.xml` (cascade) can be found in the `Data` folder

## Home-made Network Training

 Thanks to Aadhav Vignesh [2], it has been possible to create and train our own convolutional neural network using pytorch. Running for hours on our CPU for 3 epochs, we obtained a model with an accuracy greater than 90%.

 The `.pt` (learnable parameters of the network trained model) and the `history.npy` (plotting accuracy vs nb of epochs) can be found in the `Data` folder.

## Instructions to use our WebApp

* Install requirements (see below)
* Download `model_final_3_epochs.pt` and `cascadeGarbage.xml`
* Download utils, detection, templates and convolution_neural_network
* Download main.py
* Launch main.py
* Go into your favorite browser and go the url `localhost:5000`
* Enter the correct settings in the project tab and use your favorite method

You can check our demonstration in our video.

## Requirements

Please read requirements.txt and install necessary modules

To install pytorch, enter the following command :
`pip install torch==1.4.0+cpu torchvision==0.2.2 -f https://download.pytorch.org/whl/torch_stable.html`

## References

Images sources for haar cascade training :

* <https://github.com/JoakimSoderberg/haarcascade-negatives>

* <https://github.com/garythung/trashnet>

* Flickr API

Haar-Cascade classifier :

* [1] <https://amin-ahmadi.com/cascade-trainer-gui/>

* <https://medium.com/@vipulgote4/guide-to-make-custom-haar-cascade-xml-file-for-object-detection-with-opencv-6932e22c3f0e>

Deep Learning Network :

* [2] <https://www.kaggle.com/aadhavvignesh/pytorch-garbage-classification-95-accuracy>
