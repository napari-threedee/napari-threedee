# Paint labels on arbitrary planes

## Summary

Painting labels is useful for annotating voxels by semantic meaning or instance. Painting labels on planes in 3D is useful to gain full context. Additionally, the axis aligned slicing may not be optimal for viewing your structure of interest. Here we describe how to paint labels on arbitrary planes in 3D.

![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/6750b262-6c79-425f-8af5-d7459ba28a16)

## Instructions

1. To run this example, first download the script from the [Examples Gallery page](https://napari-threedee.github.io/generated/gallery/plugin/label_annotator_plugin/); the link is at the bottom of the web page. Save the file to a memorable location. Or, if you've cloned this repository, the script can be found at `napari-threedee/docs/examples/plugin/label_annotator_plugin.py`. 

2. Ensure you have activated a virtual environment with napari-threedee installed. Change directories to the script location and then execute the script using:
	```bash
	python label_annotator_plugin.py
	```  
3. The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing a 3D volume with one plane being actively rendered. The "label annotator" widget will be visible and a `Labels` layer will be present in the viewer. Note that depending on your versions of `napari-threedee` and `napari` you may see subtle visual differences between your viewer and the screenshots/videos here.

	![example opened](https://github.com/napari-threedee/napari-threedee/assets/1120672/09a6537d-882c-4b2c-95ca-e9e7ea173c8e)

4. Activate the plugin. Select the `plane` layer for the "image layer" and `Labels` for the "labels layer" and click the "activate" button.

	![activate plugin](https://github.com/napari-threedee/napari-threedee/assets/1120672/28f3f629-e62f-4d6f-8f33-c467b2cee731)

5. With the `plane` image layer selected, you can translate the rendering plane along its normal vector holding the shift key while clicking  with the left mouse button and dragging the plane. Alternately, you can re-position the render plane using [the render plane manipulator](https://napari-threedee.github.io/how_to/render_plane_manipulator/).

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/62871cf6-7f97-4808-917d-dbe7bd0e42d6)
	
6. Paint on the rendered plane. First, select the `Labels` layer in the layer list. Then click the brush icon to "Activate the paint brush" in the layer controls. Finally paint on the rendered plane by clicking with the left mouse button on the plane and dragging the mouse. You can hold the space-bar to toggle `pan` mode to allow you to change the viewing angle. You can select the `plane` image layer and reposition the rendering plane, then re-select the `Labels` layer to resume painting.

	![paint mode](https://github.com/napari-threedee/napari-threedee/assets/1120672/66d4393c-03f4-481b-bde4-d78887c1c8cb)
	
	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/6750b262-6c79-425f-8af5-d7459ba28a16)