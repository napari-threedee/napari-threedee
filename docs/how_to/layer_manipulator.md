# How to position layers with the layer manipulator

## Summary

Sometimes objects in your data have a systematic shift relative to one another. If you need to position an entire dataset, you can use the layer manipulator. This article describes how to interactively position a layer using the layer manipulator.

![type:video](https://user-images.githubusercontent.com/1120672/207429703-d595a3cc-f569-40f5-8cb8-e3fc019d983b.mov)

## Instructions

1. Open the [layer_manipulator.py](https://napari-threedee.github.io/generated/gallery/plugin/layer_manipulator_plugin/) example from the docs. The viewer is initalized in 3D rendering mode. You can 
   click and drag in the canvas to rotate the view. Note that the two layers are the same image, but not aligned.
	
	![layers loaded](https://user-images.githubusercontent.com/1120672/207427782-c2b04738-17ed-4963-b6b9-cde6a899ae7f.png)
	
2. Select the "image_1" layer in the layer manipulator plugin. We will move layer "image_1" (the green one) to be aligned with layer "image_0".

	![select layer](https://user-images.githubusercontent.com/1120672/207428382-3683481f-4acc-4f6a-a108-c497893cf82e.png)

3. Activate the layer manipulator plugin.

	![activate the plugin](https://user-images.githubusercontent.com/1120672/207428741-3d07f55f-0cc9-4b1b-92eb-6d67d7beac94.png)
	
	When the plugin is activated, you will see the manipulator at (0, 0, 0) on image_1.
	
	![plugin activated](https://user-images.githubusercontent.com/1120672/207428982-2637f2f4-7f2d-48ff-8c96-374cb6b8d7db.png)
	
4. 	Click and drag the manipulator to align the rows. Since the layers are being rendered with additive blending, the image turns white where the layers are overlapping.

	![type:video](https://user-images.githubusercontent.com/1120672/207429298-53af135e-0c5a-4517-bbb1-f97e2ccaadf7.mov)