# Annotate 3D paths

## Summary
Paths are useful for describing paths through your data. This tutorial describes how use the path annotator to interactively add paths using spline fitting to a series of points placed in a 3D image.

![type:video](https://user-images.githubusercontent.com/1120672/225981766-0e586d50-c90c-4c3e-a5e9-c5b30ba6bed2.mov)


## Instructions

1. To run this example, first download the script from the [Examples Gallery page](https://napari-threedee.github.io/generated/gallery/plugin/path_annotator_plugin/); the link is at the bottom of the web page. Save the file to a memorable location. Or, if you've cloned this repository, the script can be found at `napari-threedee/docs/examples/plugin/path_annotator_plugin.py`. 

2. Ensure you have activated a virtual environment with napari-threedee installed. Change directories to the script location and then execute the script using:
	```bash
	python path_annotator_plugin.py
	```  
	The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing two 3D volumes, each with one plane being actively rendered. Additionally, the viewer will also have a "path annotator" widget and `n3d paths` Points layer and a `n3d paths (smooth fit)` Shapes layer. Note that depending on your versions of `napari-threedee` and `napari` you may see subtle visual differences between your viewer and the screenshots/videos here.

	![example opened](https://user-images.githubusercontent.com/1120672/225978705-55570907-8f1d-4fc9-9f72-883c6c790516.png)

3. Activate the plugin. Select the `orange plane` layer for the "image layer" and click the "activate" button.

	![layer selected](https://user-images.githubusercontent.com/1120672/225978983-2a25f87d-8b7c-4e4f-906a-db0e11b75a64.png)

4. You can translate the rendered `orange plane` rendering plane along its normal vector by holding the shift key while clicking  with the left mouse button and dragging the plane.  Alternately, you can re-position the render plane using [the render plane manipulator](https://napari-threedee.github.io/how_to/render_plane_manipulator/).

	![type:video](https://user-images.githubusercontent.com/1120672/225979078-c71d1759-78e5-40c3-8304-cd71664023b7.mov)
	
5. You can begin path annotation by adding a point using alt+left mouse button clicking on the rendered plane. Ensure that the image layer `orange plane` is selected. As you alt-clicks to add points to the `n3d paths` Points layer and the annotator will automatically fit a third order spline in the `n3d paths (smooth fit)` Shapes layer. If you don't want the spline to be automatically fit, you can uncheck the "automatically fit spline" checkbox. If you are not automatically fitting the spline, you need to click the "fit spline" button to manually initiate the spline fitting. While annotating, you can move the render plane as noted in step 4 above.

	![type:video](https://user-images.githubusercontent.com/1120672/225979471-28d50f30-c97e-48a8-9349-7c776ce06427.mov)
	
5. To annotate another path, press the `n` key and repeat step 5 above. Each path is given a unique id and will have a unique color. If you want to extend a previous path, you can do so by selecting the `n3d paths` Points layer and using the "Select points" tool to select a point from a previous path. Then, reselect the image layer `orange plane` in order to place more points using alt-click.  Note the path will be extended from the end, *the last placed point for that path*, not the selected point.

	![type:video](https://user-images.githubusercontent.com/1120672/225980864-27310f9a-3fc2-4764-a4f2-909b92471edd.mov)