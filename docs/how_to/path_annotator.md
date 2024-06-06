# Annotate 3D paths

## Summary
Paths are useful for describing paths through your data. This tutorial describes how use the path annotator to interactively add paths using splines in a 3D image.

![type:video](https://user-images.githubusercontent.com/1120672/225981766-0e586d50-c90c-4c3e-a5e9-c5b30ba6bed2.mov)


## Instructions

1. To run this example, first download the script from the [Examples Gallery page](https://napari-threedee.github.io/generated/gallery/plugin/path_annotator_plugin/); the link is at the bottom. Ensure the location is somewhere memorable. Or, if you've cloned this repository, the script can be found at `napari-threedee/docs/examples/plugin/path_annotator_plugin.py`. 

2. Ensure you have the appropriate virtual environment active, change directories to the script location, and then you can execute the script using:
	```bash
	python path_annotator_plugin.py
	```  
	The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing two 3D volumes, each with one plane being actively rendered.

	![example opened](https://user-images.githubusercontent.com/1120672/225978705-55570907-8f1d-4fc9-9f72-883c6c790516.png)

2. Activate the plugin. Select the `orange` layer for the "image layer" and click the "activate" button.

	![layer selected](https://user-images.githubusercontent.com/1120672/225978983-2a25f87d-8b7c-4e4f-906a-db0e11b75a64.png)

3. You can translate the rendered orange play along its normal vector by holding the shift key while clicking  with the left mouse button and dragging the plane.

	![type:video](https://user-images.githubusercontent.com/1120672/225979078-c71d1759-78e5-40c3-8304-cd71664023b7.mov)
	
4. You can add a spline by ctrl+left mouse button clicking on the rendered plane to add points to the plane. The annotator will fit a third order spline when 4 or more points have been added. If you don't want the spline to be automatically fit, you can uncheck the "automatically fit spline" checkbox. If you are not automatically fitting the spline, you need to click the "fit spline" button to manually initiate the spline fitting.

	![type:video](https://user-images.githubusercontent.com/1120672/225979471-28d50f30-c97e-48a8-9349-7c776ce06427.mov)
	
5. You can add additional splines by advancing the "current spline index". Each spline is given a unique index. Points are added to the spline whose index is currently selected. Additionally, to make a spline that doesn't lie in a single plane, you can add points to mutiple planes by moving the rendered plane.

	![type:video](https://user-images.githubusercontent.com/1120672/225980864-27310f9a-3fc2-4764-a4f2-909b92471edd.mov)