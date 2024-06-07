# How to use the render plane manipulator

## Summary
The render plane manipulator allows you to interactively position the plane being visualized in the `napari` `Image` 
layer plane depiction mode. The plane depiction mode is a 3D rendering mode where only the specified plane is 
rendered. This is a power rendering mode for visualizing structures within a 3D volume, but defining the precise 
plane programatically can be challenging. Thus, the render plane manipulator is useful for interactively positioning 
the rendering plane.

![type:video](https://user-images.githubusercontent.com/1120672/207312303-e81f652a-3fae-476f-abee-e19227b2b6c3.mov)


## Instructions

1. Launch `napari`. Open your terminal and activate your `napari-threedee` environment. Enter `napari` in the 
   command prompt to launch napari.

     ```bash
     napari
     ```

2. Load the "HIV virus-like particles tomogram" image from the File menu: **File -> Open Sample -> HIV virus-like particles tomogram**. Note that this will download the ~470 Mb sample image from Zenodo([https://doi.org/10.5281/zenodo.6504891](https://doi.org/10.5281/zenodo.6504891)) so it make take some time. If you have previously opened the sample image, it should be cached and will not be downloaded again.

	![screenshot of the hiv tomogram](https://user-images.githubusercontent.com/1120672/207310777-1cfdb146-e5b9-43fb-a740-6af137ed9df5.png)

3. Click the "Toggle ndisplay" button in the lower left corner of the viewer to enter 3D rendering mode. 

	![toggle ndisplay button](https://user-images.githubusercontent.com/1120672/207310915-45424cd4-a0c6-44e9-9de1-93483959a131.png)

	Upon clicking the button, you will see the volume in 3D. You you can click and drag to rotate the view. Note that the HIV particles are visible, but low contrast.
	
	![particles rendered in 3D](https://user-images.githubusercontent.com/1120672/207311476-e5e8d2dd-61b1-46f3-8607-e22c3da9afbb.png)

4. In the layer controls, change the `depiction` mode from `volume` to `plane`.

	![depiction mode dropdown.](https://user-images.githubusercontent.com/1120672/207311566-8da18aa4-8b65-40b7-925a-bd679e36ff82.png)
	
	After changing the depiction mode, you will see a single plane being rendered from your 3d volume.
	
	![plane rendering mode](https://user-images.githubusercontent.com/1120672/207311620-0494f9cf-3059-40d4-902e-37f026556c56.png)

5. Open the render plane manipulator widget using the menu: **Plugins -> napari-threedee -> render plane manipulator**. You will see the widget appear on the right side of the napari viewer. Ensure that "HIV tomogram" is selected in the "layers" dropdown box and then click the "activate" button to activate the manipulator function. A render plane manipulator will appear in the corner of the rendering plane. 

6. You can double-click with the left mouse button (or hold down the Shift key while clicking) to reposition the manipulator. Note that depending on your versions of `napari-threedee` and napari there may be a subtle visual difference between what you see in `napari` and the screenshots/videos here.

	![render plane manipulator](https://user-images.githubusercontent.com/1120672/207311868-d6a0d972-37ea-4e79-92b1-3923a058221b.png)
	

7. You can click and drag the translator on the manipulator to translate the plane along its normal.

	![type:video](https://user-images.githubusercontent.com/1120672/207312152-d9d49bfd-04dc-4b27-827b-04282c512e48.mov)

8. You can click and drag the rotator to rotate the plane around the manipulator.

	![type:video](https://user-images.githubusercontent.com/1120672/207312303-e81f652a-3fae-476f-abee-e19227b2b6c3.mov)

	If you would like to change the center of rotation, you can double-click with the left mouse button (or hold down the Shift key while clicking) on the plane with the left mouse button to move the manipulator.
	
	![type:video](https://user-images.githubusercontent.com/1120672/207312430-74b95837-0718-4b9b-a2dd-b9fed0565e21.mov)