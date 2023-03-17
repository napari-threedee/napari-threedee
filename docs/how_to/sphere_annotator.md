# Annotate spheres on arbitrary planes

## Summary
Annotating spheres in 3D is useful for defining regions of interest around a point. Here we describe how to annotate a sphere on an arbitrary plane in an image.

![type:video](https://user-images.githubusercontent.com/1120672/225970403-f75e90dc-614d-409d-b1b7-f7453dededc1.mov)


## Instructions
1. Open the `sphere_annotator.py` example from the examples folder. The viewer is initialized in 3D rendering mode in plane rendering mode. We are viewing a 3D volume with one plane being actively rendered.

	![example opened](https://user-images.githubusercontent.com/1120672/225966223-fb426704-efb3-4ead-a7a5-8d0731f890ff.png)

2.  Activate the plugin. Select the `plane` layer for the "image layer" and click the "activate" button. This will add points to the `Points` layer based on the intersection of the click ray with the currently rendered `orange` layer.

	![select layer](https://user-images.githubusercontent.com/1120672/225966635-337d550d-6880-424a-8b3b-d0f1de07b42f.png)
	
3. You can translate the rendered orange play along its normal vector by holding the shift key while clicking  with the left mouse button and dragging the plane.

	![type:video](https://user-images.githubusercontent.com/1120672/225967164-978493a3-ee48-4768-8359-029b6b18bb3f.mov)
	
4. You can add a sphere on the rendered play by holding alt and clicking on the plane with the left mouse button. If you alt+click again, it will reposition the sphere.

![type:video](https://user-images.githubusercontent.com/1120672/225969195-c892ed99-0d97-45bf-b701-4ec07df90050.mov)

5. You can adjust the radius of the sphere by presing the `r` key and the sphere will snap to the position where your mouse pointer intersects the rendered plane.

![type:video](https://user-images.githubusercontent.com/1120672/225968729-ba5c2d6b-d2a3-4012-9c71-dc8ef9abb4a3.mov)

6. You add a new point by pressing the `n` key and repeating steps 5 and 6.

![type:video](https://user-images.githubusercontent.com/1120672/225969966-c6ba56d0-4f27-4ad2-977d-b07e26b302f5.mov)