## <b>Artistic applications of AutoEncoders - Colorization</b> <br>

**[Sept20]** This is the result of the work done for the Artificial Intelligence with Deep Learning postgraduate course at the UPC (Ed. 2019-2020).

The goal of the experiment is to explore and understand the filtering, denoising, colorization, etc. of images, but with focus on colorization of face images: 

**Grayscale to Colored Images**

![Colorization concept image](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Colorization_concept.png)

**Milestones in the colorization experiment:**

Build a face images dataset based on Voxceleb2 - Exploring a vanilla AE - Using AE for face colorization

**.-Building the Dataset**

Development of a script to generate a subset of 110K face images by picking just a few frames from videos of Voxceleb2 dataset.

The script can be found in the following link:
[[Get Images Script]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Get_Images.ipynb) <br>

The generated dataset can be downloaded from the following link:
[[Generated Dataset]](https://drive.google.com/drive/folders/1tRzBwu84J3xty2zPY3RU3rtYEppL3a3I?usp=sharing)

**.-Exploring a Vanilla AutoEncoder**

This task consists in the development of a vanilla AutoEncoder with a basic architecture of 3 downsampling steps, 1 linear fc layer and 3 upsamplig steps.

The task has been developed for a better understanding of AutoEncoders, architectures, filters learned, outputs etc...

The jupyter notebook used for this task can be found here: 
[[Vanilla AutoEncoder]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Convolutional_Autoencoder_complete.ipynb)


**.-Using AutoEncoders for face colorization**

This task consists in the development of an AutoEncoder for colorizing faces. The colorization strategy has been taken from the paper Zhang, R., Isola, P., & Efros, A. A. [[Colorful image colorization.]](https://arxiv.org/abs/1603.08511) ECCV 2016.

For the colorization of images we’ll use the Lab color space that completely separates the lightness from color. The Lab color space allows to get pure black and white (grayscale) information of an image in one single channel (L) and the other two channels (ab) will contain the color information. For our dataset we need to perform a preprocessing by transforming the RGB images into Lab images, feed the AE with (L) channel as input and corresponding (ab) channels as the supervisory signal and finally postprocess the output to save the Lab image into a RGB again.
![Lab_Space](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Lab_Space.png)

Two approaches has been developed to check the performance in colorization:

   **1st approach: Train from scratch an AE using our own dataset**

Encoder consisting of some convolutional layers with ReLU activation function and decoder with upsampling layers to restore dimensions:
![Architecture 1st](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Architecture_1st.png)

   **2nd approach: Using transfer learning technique**

Connecting a VGG16 pretrained model as a feature extractor (removing the classification part) with the decoder.:
![Vgg16](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Vgg16.png)

The jupyter notebook used for this task can be found here: 
[[Face colorization]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Colorization_05_Adam_mse%2BTransfer_Learning.ipynb)

**Colorization experiments and some conclusions:**

In general, achieved face image colorization is reasonable but can be improved with bigger dataset. 

The approach 2 using transfer learning achieves visually and numerically a better performance with lower computational resources:

Accuracy approach 1: 0.8747 and uses 6,219,538 trainable params

Accuracy approach 2: 0.8913 and uses 1,572,114 trainable params 
![Results_Colorization](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Results_Colorization.png)
The colorization of other kind of images is failing due to face biased dataset: 
![Landscape Colorization](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/lanscape.jpg)
