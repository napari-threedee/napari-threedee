# How to use the clipping plane manipulator

## Summary
The clipping plane manipulator allows you to interactively position a clipping plane for a layer being visualized in the `napari`. Clipping planes can be used in 3D rendering mode to only show part of the volume. This is a power rendering mode for visualizing structures within a 3D volume, but `napari` currently does not have controls for precise 
positioning of planes in the graphical user interface. Thus, the clipping plane manipulator is useful for interactively positioning 
clipping planes.

![type:video](https://github.com/user-attachments/assets/df7347d7-f0c4-4e53-ac8c-cfa0dfed11af)


## Instructions

1. Launch `napari`. Open your terminal and activate your `napari-threedee` environment. Enter `napari` in the 
   command prompt to launch napari.

     ```bash
     napari
     ```

2. Load the "Cells (3D+2Ch)" `napari` sample image from the File menu: **File -> Open Sample -> napari builtins -> Cells (3D+2Ch)**.

3. Click the "Toggle ndisplay" button in the lower left corner of the viewer to enter 3D rendering mode. 

	Upon clicking the button, you will see the volume in 3D. You you can click and drag to rotate the view and use the scrool wheel to zoom in/out. Rotate the view to look at the volume at a more oblique angle and zoom out a little.

	![screenshot in 3D](https://github.com/user-attachments/assets/89b68026-37d5-4b7a-8184-7db58e94f79a)

5. Open the clipping plane manipulator widget using the menu: **Plugins -> napari-threedee -> clipping plane manipulator**. You will see the widget appear on the right side of the napari viewer. Ensure that "membrane" is selected in the "layers" dropdown box and then click the "activate" button to activate the manipulator function. A clipping plane manipulator will appear in the corner of the volume plane. Note: you can hide the widget by clicking the crossed eye icon at the top left of the widget.

6. Select the layer named "membrane" in the layer list on the left. You can now double-click with the left mouse button to reposition the manipulator. Note that depending on your versions of `napari-threedee` and napari there may be a subtle visual difference between what you see in `napari` and the screenshots/videos here.

	![clipping plane manipulator](https://github.com/user-attachments/assets/0b7452f1-8802-4063-8cc7-0f2cc42a399e)
	

7. You can click and drag the translator on the manipulator to translate the plane along its normal. You can click and drag the rotator to rotate the orientation of the manipulator. If you would like to change the center of rotation, you can double-click with the left mouse button to reposition the manipulator.

	![type:video](https://github.com/user-attachments/assets/df7347d7-f0c4-4e53-ac8c-cfa0dfed11af)
