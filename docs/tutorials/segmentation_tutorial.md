# Interactive, GPU-accelerated 3D segmentation

## Summary

In this tutorial, we will learn how to combine two napari-threedee plugins with `napari-accelerated-pixel-and-object-classification` (napari-apoc) to perform interactive 3D segmentation of nuclei. This demonstrates how the interoperatibility of napari-threedee plugins allows users to create human in the loop workflows. You will use the render plane manipulator plugin to view arbitrary slices of a 3D image of nuclei and then label annotator to paint labels in 3D for ground truth annotations for the napari-apoc segmentation algorithm. We use the napari-apoc GPU-accelerated random forrest classifier to predict which pixels in the image belong to nuclei usign only a few sparse labels.

![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/6750b262-6c79-425f-8af5-d7459ba28a16)

## Prerequisites
To perform this tutorial, you need the following:

1. Basic familiarity with napari. 
2. A computer with a working conda/mamba installation. We will use conda/mamba to manage the environment we install the dependencies into.  

## Setup
Before trying the tutorial, please initialize your python environment as described below. Note that we recommend using `mamba` to install/manage dependencies. If you are using `conda`, you can simply replace `mamba` with `conda` in the commands below, but we recommend you that the conda version is >23.10

1. Create a new environment called `seg-env` with python 3.9. In your terminal, enter the following command.

	```bash
	mamba create -n seg-env -c conda-forge python=3.9
	```

2. Activate your new environment. If you successfully activated the environment, you should see the name of your environment (`(seg-env)`) to the left of your prompt.

	```bash
	mamba activate seg-env
	```

3. Install the dependencies.

	```bash
	mamba install -c conda-forge napari pyqt napari-threedee napari-accelerated-pixel-and-object-classification
	```

4. If your are on Mac OS, install this openCL wrapper (used by APOC for GPU acceleration):

	```bash
	mamba install -c conda-forge ocl_icd_wrapper_apple
	```

5. If you are on linux, install this openCL wrapper (used by APOC for GPU acceleration):
	
	```bash
	mamba install -c conda-forge ocl-icd-system
	```


## Instructions

1. Activate the environment you created during the setup above. If you successfully activated the environment, you should see the name of your environment (`(seg-env)`) to the left of your prompt.

	```bash
	mamba activate seg-env
	```

2. Launch napari from the command line using the command below. The first time you launch napari, it may take a few moments to open.

	```bash
	napari
	```
	
	![napari opened](https://github.com/napari-threedee/napari-threedee/assets/1120672/7e59814f-b9ee-40b1-88f6-03ed5ae7adb0)

3. Open the 3D cells sample image from the File menu. **File -> Open Sample -> napari builtins -> Cells (3D+2Ch)**. This will load two image layers: DAPI-stained nuclei and the cell membranes.

	![sample data](https://github.com/napari-threedee/napari-threedee/assets/1120672/cd954faf-50d2-4f5c-a38d-f87868ed5e8c)

4. Delete the membrane layer. First select the membrane layer from the layer list and then click the "garbage can" button.

	![delete membranes](https://github.com/napari-threedee/napari-threedee/assets/1120672/e638fd3f-686a-4638-a17b-c0c48ee1b7e9)

5. Create a new label layer by clicking the new label layer button above the layer list. This will add a new layer called "Label".

	![new labels](https://github.com/napari-threedee/napari-threedee/assets/1120672/4ad191be-50bb-4896-aeac-65e35f017720)

6. Switch to 3D rendering mode. Click the rending toggle (square button in the lower left corner of the viewer. You can scroll up/down in the canvas to zoom in/out and click and drag to rotate the view.

	![rendering toggle](https://github.com/napari-threedee/napari-threedee/assets/1120672/176e7481-7eb0-4fdc-87f9-c120925fafd7)
	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/66891666-a639-4ad0-aba0-773888790f48)	
7. Switch the depiction to "plane". This will render only one plane from the 3D volume. First select the "nuclei" layer from the layer list. Then change the depiction to "plane" in the layer controls.

	![plane depiction](https://github.com/napari-threedee/napari-threedee/assets/1120672/22d3290b-5afc-42f0-b566-e3b01249a1d8)

8. Use the napari-threedee render plane manipulator to move the rendering plane and explore the image. Open the render plane manipulator from the plugin menu: **Plugins -> napari-threedee -> render plane manipulator**. Click the "activate" button the start the manipulator. Then use it to rotate the rendering plane to find a plane of interest. Note: you can double-click with the mouse to re-position the manipulator on the plane.

	![plane manipulator](https://github.com/napari-threedee/napari-threedee/assets/1120672/db651930-a6f3-4329-8ec7-50a01b701093)
	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/2f0b7490-9281-4a19-b59f-7c189975ad37)

9. Next, we will use the label annotator to paint labels directly on the rendering plane. We will use these labels to train a random forest classifier to segment the nuclei. Open the label annotator plugin from the plugins menu: **Plugins -> napari-threedee -> label annotator**. Activate the label annotator by clicking the "activate" button.

	![label annotator](https://github.com/napari-threedee/napari-threedee/assets/1120672/58ccdf39-97ef-4f1a-af1f-9d86dac402e8)
	
10. Switch to painting mode. First select the labels layer by clicking on the "Label" layer in the layer list. Then click on the paint brush icon in the layer controls.

	![paint mode](https://github.com/napari-threedee/napari-threedee/assets/1120672/2874ffd0-2f77-408f-8690-e336b2edff1b)
	
11. We will label the image as either background or nuclei so that we can train the random forest classifier to predict if each pixel is belongs to the background or a nucleus. We will start by labeling some pixels as background by giving them the value of 1. Random forest classifiers can be trained with small numbers of annotations. You can label some background pixels by clicking and dragging on the rendered plane.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/03df1aac-31c3-4b5f-bfde-5d0b9dcacf72)
	
12. Next, we will add some nuclei labels. First change the label index from 1 to 2 in the layer controls (upper left hand corner of the window). Then paint some of the nuclei. Like with the background, a couple of lines is sufficient.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/9d3257c9-8b3a-4d18-ba9d-94b1569521c9)
	
13. We can also add labels to other planes. To do so, select the "nuclei" layer and use the render plane manipulator to select a new plane. Then re-select the "Label" layer and paint as you did before. Remember that label value 1 is used for background and label value 1 is used for nuclei.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/1d75bd05-ca03-4ba9-a0f6-2608442e5939)

14. Finally, we can train the random forest classifier. Open the napari-APOC semantic segmentation plugin: **Plugins -> napari-accelerated-object-and-pixel-classification -> Semantic Segmentation**. Note: for the other two widgets, you can click the middle, eye icon to hide them. First, select the "nuclei" layer in the "Select images (channels) used for training" box at the top. Then select the "Label" layer in the "Select ground truth annotation" combobox in the lower portion of the widget. Finally, click the "Train" button—we will leave the other parameters as defaults. When the training has completed, the result will be added as a new layer called "Result of PixelClassifier". Notice that it is hard to see the segmented nuclei because both the background and nuclei predictions are shown together.

	![semantic segmentation](https://github.com/napari-threedee/napari-threedee/assets/1120672/11c953d5-aa55-443a-a5ce-26b1b65a4204)
	![segmentation result](https://github.com/napari-threedee/napari-threedee/assets/1120672/8f7ed183-99a1-4e48-bec0-3422211bd449)
	
15.  We can view just the predicted nuclei labels by selecting the nucleus label value (2) in the label box in the layer controls and then clicking the "show selected" checkbox. Note that we have obtained a pretty good segmentation with just a few labels. You can also compare the segmentation result to the raw image by toggling visibility of the different layers.
	
	![just nuclei](https://github.com/napari-threedee/napari-threedee/assets/1120672/1f1f98cc-4de3-4bfd-a74d-ada03eac87f2)
	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/aebe3a8c-0a95-499c-8194-4d406423f9f3)

16. If you are not happy with your segmentation, you can return to the labeling and training steps above to add more labels and improve the segmentation algorithm.
