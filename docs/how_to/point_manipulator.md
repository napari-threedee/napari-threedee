# How to position points with the point manipulator

## Summary
Precisely positioning a point in 3D is a useful way to annotate specific positions or choose the location of an object in a scene. This article explains how to use the `napari-threedee` point manipulator plugin to position points in 3D.

![type:video](https://user-images.githubusercontent.com/1120672/207435568-f4a2afd9-28e9-481c-97aa-6f8994382834.mov)


## Instructions

1. Open the [points_manipulator.py](https://napari-threedee.github.io/generated/gallery/plugin/points_manipulator_plugin/) example.


    ![example opened](https://user-images.githubusercontent.com/1120672/207381513-8fd3f7e9-157f-4ef9-99c1-551bd329842b.png)
   The example opens in 3D rendering mode. Click and drag to rotate the view.
   
2. Open the point manipulator plugin from the Plugins menu: Plugins -> napari-threedee -> point manipulator. Click the "activate" to start the plugin.

	![plugin opened](https://user-images.githubusercontent.com/1120672/207382282-dad2bd6f-68cf-47d6-89ed-be326d320f93.png)

4. Set the Points later to "Selection" mode by clicking the "Selection" button in the layer controls

	![point selection mode](https://user-images.githubusercontent.com/1120672/207382556-9cd2111a-1a01-4102-9de8-4bb87ddad3c3.png)

5. Select a point by clicking on it. The point manipulator will appear on the selected point.

	![point manipulator](https://user-images.githubusercontent.com/1120672/207383241-d86cdee0-8f5f-4e0d-bb96-bebb993e3904.png)
	
6. You can now move the point around by clicking and dragging the manipulator.

	![type:movie](https://user-images.githubusercontent.com/1120672/207384092-1b4231fa-beba-46e4-b1c7-b32aa1ae32a6.mov)
	
	If you want to rotate the view while a point manipulator is active, press and hold the space bar while clicking and dragging in the canvas to rotate the view. The manipulator will be rendered invisible while you rotate so you can see the whole scene.
	
	![type:movie](https://user-images.githubusercontent.com/1120672/207384729-fea3e148-a61a-43c0-bc78-eeb8f973e36b.mov)