# Paint labels on arbitrary planes

## Summary

Painting labels is useful for annotating voxels by semantic meaning or instance. Painting labels on planes in 3D is useful to gain full context. Additionally, the axis aligned slicing may not be optimal for viewing your structure of interest. Here we describe how to paint labels on arbitrary planes in 3D.

![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/6750b262-6c79-425f-8af5-d7459ba28a16)

## Instructions

1. Open the `plane_labeler_plugin.py` example from the examples folder. The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing a 3D volume with one plane being actively rendered.

	![example opened](https://github.com/napari-threedee/napari-threedee/assets/1120672/09a6537d-882c-4b2c-95ca-e9e7ea173c8e)

2. Activate the plugin. Activate the plugin. Select the `plane` layer for the "image layer" and `Label` for the "labels layer" and click the "activate" button.

	![activate plugin](https://github.com/napari-threedee/napari-threedee/assets/1120672/28f3f629-e62f-4d6f-8f33-c467b2cee731)

3. You can translate the plane along its normal vector by selecting the `plane` image layer in the layer list and holding the shift key while clicking  with the left mouse button and dragging the plane.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/62871cf6-7f97-4808-917d-dbe7bd0e42d6)
	
4. Paint on the rendered plane. First, select the labels layer in the layer list. Then switch to "painting mode" on the layer controls. Finally paint on the rendered plane by clicking with the left mouse button on the plane and dragging the mouse.

	![paint mode](https://github.com/napari-threedee/napari-threedee/assets/1120672/66d4393c-03f4-481b-bde4-d78887c1c8cb)
	
	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/6750b262-6c79-425f-8af5-d7459ba28a16)