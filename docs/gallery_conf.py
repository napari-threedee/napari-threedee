from pathlib import Path
import warnings

from mkdocs_gallery.scrapers import figure_md_or_html, matplotlib_scraper
from mkdocs_gallery.gen_data_model import GalleryScript
import napari
import qtgallery

warnings.filterwarnings("ignore", category=DeprecationWarning)


def napari_image_scraper(block, script: GalleryScript):
    """Scrape screenshots from napari windows.

    Parameters
    ----------
    block : tuple
        A tuple containing the (label, content, line_number) of the block.

    script : GalleryScript
        Script being run

    Returns
    -------
    md : str
        The ReSTructuredText that will be rendered to HTML containing
        the images. This is often produced by :func:`figure_md_or_html`.
    """
    viewer = napari.current_viewer()
    if viewer is not None:
        image_path = next(script.run_vars.image_path_iterator)
        screenshot = viewer.screenshot(canvas_only=False, flash=False, path=image_path)
        viewer.close()
        return figure_md_or_html([image_path], script)
    else:
        return ""


def _reset_napari(gallery_conf, file: Path):
    # Close all open napari windows and reset theme
    while napari.current_viewer() is not None:
        napari.current_viewer().close()
    settings = napari.settings.get_settings()
    settings.appearance.theme = 'dark'
    # qtgallery manages the event loop so it
    # is not completely blocked by napari.run()
    qtgallery.reset_qapp(gallery_conf, file)


conf = {
    "image_scrapers": [napari_image_scraper, matplotlib_scraper],
    "reset_modules": [_reset_napari],
}
