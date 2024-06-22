# Annotate points on an arbitrary plane

## Summary
Annotating points in 3D can be useful for picking particles in CryoET and annotating locations of landmarks. However, doing so on a 2D computer screen is challenging. Here we demonstrate how to use the point annotator, which annotates points on arbitrary slicing planes.

![type:video](https://user-images.githubusercontent.com/1120672/225956459-2da07c2d-b7c4-4e61-aa7e-11d11cc06e79.mov)

## Instructions

1. To run this example, first download the script from the [Examples Gallery page](https://napari-threedee.github.io/generated/gallery/plugin/point_annotator_plugin/); the link is at the bottom of the web page. Save the file to a memorable location. Or, if you've cloned this repository, the script can be found at `napari-threedee/docs/examples/plugin/point_annotator_plugin.py`. 

2. Ensure you have activated a virtual environment with napari-threedee installed. Change directories to the script location and then execute the script using:
	```bash
	python point_annotator_plugin.py
	``` 

3. The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing two 3D volumes, each with one plane being actively rendered. Note that depending on your versions of `napari-threedee` and `napari` you may see subtle visual differences between your viewer and the screenshots/videos here.

	![example opened](https://user-images.githubusercontent.com/1120672/225954648-08094784-e538-4bdb-9d5e-2f39b3770a16.png)

4. Activate the plugin. Select the `orange plane` layer for the "image layer" and `Points` for the "points layer". Finally click the "activate" button. This will add points to the `Points` layer based on the intersection of the click ray with the currently rendered `orange plane` layer.

	![select layer](https://user-images.githubusercontent.com/1120672/225953774-cf391f47-b769-493d-8e94-caa5f92ebe6a.png)

5. You can translate the `orange plane` layer render plane along its normal vector by holding the shift key while clicking with the left mouse button and dragging the plane. Alternately, you can re-position the render plane using [the render plane manipulator](https://napari-threedee.github.io/how_to/render_plane_manipulator/).

	![type:video](https://user-images.githubusercontent.com/1120672/225954237-e77891ee-2302-47f6-ad14-963da44f8dac.mov)
	
6. You can add points on the `orange plane` layer rendered plane, at the location of your mouse cursor, by pressing the `a` key or by holding the *Alt* key while clicking with the left mouse button. Note you must ensure that the `orange plane` layer is selected, the points will be automatically placed in the `Points` layer. You can move the plane to annotate a new plane by clicking and dragging with the shift key held down as in step 5.

	![type:video](https://user-images.githubusercontent.com/1120672/225955156-5c05c7a4-814c-4aa0-89b0-9dee8713e8c3.mov)
