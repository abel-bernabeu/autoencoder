<!--<h3><b>Artistic applications of AutoEncoders</b></h3>-->
## <b>Artistic applications of AutoEncoders - Colorization</b> <br>

**[Sept20]** This is the result of the work done for the Artificial Intelligence with Deep Learning postgraduate course at the UPC (Ed. 2019-2020).

The goal of the experiment is to explore and understand the filtering, denoising, colorization, etc. of images, but with focus on colorization of face images: 

**Grayscale to Colored Images**

![Colorization concept image](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Colorization_concept.png)

**Milestones in the colorization experiment:**

Build a face images dataset based on Voxceleb2 - Exploring a vanilla AE - Using AE for face colorization

**Building the Dataset**

Development of a script to generate a subset of 110K face images by picking just a few frames from videos of Voxceleb2 dataset.

The script can be found in the following link:
[[Get Images Script]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Get_Images.ipynb) <br>

The generated dataset can be downloaded from the following link:
[[Generated Dataset]](https://drive.google.com/drive/folders/1tRzBwu84J3xty2zPY3RU3rtYEppL3a3I?usp=sharing)

**Exploring a Vanilla AutoEncoder**

This task consists in the development of a vanilla AutoEncoder with a basic architecture of 3 downsampling steps, 1 linear fc layer and 3 upsamplig steps.

The task has been developed for a better understanding of AutoEncoders, architectures, filters learned, outputs etc...

The jupyter notebook used for this task can be found here: 
[[Vanilla AutoEncoder]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Convolutional_Autoencoder_complete.ipynb)


**Using AutoEncoders for face colorization**

This task consists in the development of an AutoEncoder for colorizing faces. The colorization strategy has been takeen from the paper Zhang, R., Isola, P., & Efros, A. A. [[Colorful image colorization.]](https://arxiv.org/abs/1603.08511) ECCV 2016.

For the colorization of images we’ll use the Lab color space that completely separates the lightness from color. The Lab color space allows to get pure black and white (grayscale) information of an image in one single channel (L) and the other two channels (ab) will contain the color information. For our dataset we need to perform a preprocessing by transforming the RGB images into Lab images, feed the AE with (L) channel as input and corresponding (ab) channels as the supervisory signal and finally postprocess the output to save the Lab image into a RGB again.

![Lab_Space](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Lab_Space.png)

The jupyter notebook used for this task can be found here: 
[[Face colorization]](https://github.com/abel-bernabeu/autoencoder/blob/master/colorization/Colorization_05_Adam_mse%2BTransfer_Learning.ipynb)
