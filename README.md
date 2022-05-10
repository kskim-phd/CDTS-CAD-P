# CDTS-CAD

# AI-based computer-aided diagnostic system of chest digital tomography synthesis: Demonstrating comparative advantage with X-ray-based AI systems

We release CDTS-CAD evaluation code.

Collaborators: Kyung-su Kim*, Juhwan Lee*, Seongje Oh* (Equal contribution)

Detailed instructions for testing the image are as follows.

------

# Implementation

A PyTorch implementation of DTS-CAD based on original pytorch-gradcam code.

pytorch-gradcam[https://github.com/jacobgil/pytorch-grad-cam] (Thanks for Jacob Gildenblat and contributors.)

------
## Environments

The setting of the virtual environment we used is described as packagelist.txt.

------
## Multi view dataset

Please downloading the our multi view dataset [here](https://drive.google.com/file/d/15vYbw43A9DXF7IXPaQxMNXSFiKy5N8UK/view?usp=sharing)

------
## N/A diagnosis (5/2)

please downloading the pre-trained weight file [here](https://drive.google.com/file/d/198TmyO5YtXlO-Acn5VE16n_52s5bscSb/view?usp=sharing). 
Please run "Classification/N_A_inference.py"

```
python N_A_inference.py 
```
You will see result of baseline and proposed(N/A)

------
## Segmentation

Put the test data in the "dataset" folder to create a split mask. please downloading the pre-trained weight file [here](https://drive.google.com/file/d/1Mqs8HA8vjrClaVNMvUbEL__cPPm90scX/view?usp=sharing).  
Please run "Segmentation/mask_maker.py".

```
python mask_maker.py 
```
The segment mask (file name : same name+"mask.jpg") is stored in the same folder.

------

## GradCAM

Please run "visualize_gradcam.py"

```
python visualize_gradcam.py
```

If you run four things sequentially, you will see that a "Result_visualize" folder is created, storing the GradCAM

------

