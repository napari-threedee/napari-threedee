# Annotate points on an arbitrary plane

## Summary
Annotating points in 3D can be useful for picking particles in CryoET and annotating locations of landmarks. However, doing so on a 2D computer screen is challenging. Here we demonstrate how to use the point annotator, which annotates points on arbitrary slicing planes.

![type:video](https://user-images.githubusercontent.com/1120672/225956459-2da07c2d-b7c4-4e61-aa7e-11d11cc06e79.mov)

## Instructions

1. Open the `plane_point_annotator.py` example from the examples folder. The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing two 3D volumes, each with one plane being actively rendered.

	![example opened](https://user-images.githubusercontent.com/1120672/225954648-08094784-e538-4bdb-9d5e-2f39b3770a16.png)

2. Activate the plugin. Select the `orange` layer for the "image layer" and `Points` for the "points layer". Finally click the "activate" button. This will add points to the `Points` layer based on the intersection of the click ray with the currently rendered `orange` layer.

	![select layer](https://user-images.githubusercontent.com/1120672/225953774-cf391f47-b769-493d-8e94-caa5f92ebe6a.png)

3. You can translate the rendered orange play along its normal vector by holding the shift key while clicking  with the left mouse button and dragging the plane.

	![type:video](https://user-images.githubusercontent.com/1120672/225954237-e77891ee-2302-47f6-ad14-963da44f8dac.mov)
	
4. You can add points on the rendered plane by holding the alt key while clicking with the left mouse button. You can move the plane to annotate a new plane by clicking and dragging with the shift key as in step 3.

	![type:video](https://user-images.githubusercontent.com/1120672/225955156-5c05c7a4-814c-4aa0-89b0-9dee8713e8c3.mov)