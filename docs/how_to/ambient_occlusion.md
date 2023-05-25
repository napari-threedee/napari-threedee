# Add ambient occlusion to surfaces

## Summary
[Ambient occlusion](https://en.wikipedia.org/wiki/Ambient_occlusion) is a rendering technique that can be used to Adding ambient occlusion can help improve the depth in the image and make detailed surfaces more clear. In this tutorial, you will learn how to apply ambient occlusion to a napari surface layer using the "ambient occlusion controls" plugin.

![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/39b877b3-5242-4229-b154-6d56dd109a5b)


## Instructions

1. Open the [`ambient_occlusion_plugin.py` example](https://napari-threedee.github.io/generated/gallery/plugin/ambient_occlusion_plugin/). The viewer is initialized in 3D rendering mode and should display a mesh of a triceratops.

	![example loaded](https://github.com/napari-threedee/napari-threedee/assets/1120672/5ae66540-4081-4885-99d3-5f9447a3f92b)

2. Select the "triceratops" layer from the plugin selection box and click the "update ambient occlusion" button. This should complete quickly for the triceratops mesh. For larger meshes, this will take longer.

	![select_layer](https://github.com/napari-threedee/napari-threedee/assets/1120672/83ab0f27-5abb-48b6-8e7d-6b3b4cd92fbc)

3. You should now see the triceratops has shadows in regions that would be blocked from ambient light.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/39b877b3-5242-4229-b154-6d56dd109a5b)
	
	
4. You can deactivate the ambient occlusion by de-selecting the layer in the plugin selection box and clicking the "update ambient occlusion" button.

	![type:video](https://github.com/napari-threedee/napari-threedee/assets/1120672/eb8f2602-e1d1-4f72-b353-275d57363c3c)