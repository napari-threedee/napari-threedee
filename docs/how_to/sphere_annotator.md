# Annotate spheres on arbitrary planes

## Summary
Annotating spheres in 3D is useful for defining regions of interest around a point. Here we describe how to annotate a sphere on an arbitrary plane in an image.

![type:video](https://user-images.githubusercontent.com/1120672/225970403-f75e90dc-614d-409d-b1b7-f7453dededc1.mov)


## Instructions
1. To run this example, first download the script from the [Examples Gallery page](https://napari-threedee.github.io/generated/gallery/plugin/sphere_annotator_plugin/); the link is at the bottom of the web page. Save the file to a memorable location. Or, if you've cloned this repository, the script can be found at `napari-threedee/docs/examples/plugin/sphere_annotator_plugin.py`. 

2. Ensure you have activated a virtual environment with napari-threedee installed. Change directories to the script location and then execute the script using:
	```bash
	python sphere_annotator_plugin.py
	``` 
3. The viewer is initialized in 3D rendering mode with an image layer `plane` in plane rendering mode. We are viewing a 3D volume with one plane being actively rendered. Additionally, the viewer will also have a "sphere annotator" widget and `n3d spheres` Points layer and a `sphere meshes` Surface layer. Note that depending on your versions of `napari-threedee` and `napari` you may see subtle visual differences between your viewer and the screenshots/videos here and the layers may have slightly different names.

	![example opened](https://user-images.githubusercontent.com/1120672/225966223-fb426704-efb3-4ead-a7a5-8d0731f890ff.png)

4.  Activate the plugin. Select the `plane` layer for the "image layer" and click the "activate" button.

	![select layer](https://user-images.githubusercontent.com/1120672/225966635-337d550d-6880-424a-8b3b-d0f1de07b42f.png)
	
5. You can translate the rendered orange plane along its normal vector by holding the shift key while clicking with the left mouse button and dragging the plane.  Alternately, you can re-position the render plane using [the render plane manipulator](https://napari-threedee.github.io/how_to/render_plane_manipulator/).

	![type:video](https://user-images.githubusercontent.com/1120672/225967164-978493a3-ee48-4768-8359-029b6b18bb3f.mov)
	
6. You can add a sphere on the rendered play by holding alt and clicking on the plane with the left mouse button. Note you must ensure that the `plane` layer is selected. A blue point marking the centroid will be placed on the `n3d spheres` Points layer and a mesh for the sphere will be created on the `sphere meshes` Surface layer. If you alt+click again, it will reposition the center of the sphere.

![type:video](https://user-images.githubusercontent.com/1120672/225969195-c892ed99-0d97-45bf-b701-4ec07df90050.mov)

7. You can adjust the radius of the sphere by pressing the `r` key: the radius of the sphere mesh will snap to the position where your mouse pointer intersects the rendered plane.

![type:video](https://user-images.githubusercontent.com/1120672/225968729-ba5c2d6b-d2a3-4012-9c71-dc8ef9abb4a3.mov)

8. You add a new sphere by pressing the `n` key and repeating steps 6 and 7. Note that the point marking the centroid will have a new color, but the initial sphere mesh will have the same radius as previously.

![type:video](https://user-images.githubusercontent.com/1120672/225969966-c6ba56d0-4f27-4ad2-977d-b07e26b302f5.mov)