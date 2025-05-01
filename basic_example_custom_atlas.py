import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neutil import Neutil

script_dir = os.path.dirname(os.path.abspath(__file__))

pnt = Neutil(
    segmentation_folder=os.path.join(
        script_dir, "./tests/test_data/nonlinear_allen_mouse/segmentations/"
    ),
    alignment_json=os.path.join(
        script_dir, "./tests/test_data/nonlinear_allen_mouse/alignment.json"
    ),
    colour=[0, 0, 0],
    atlas_path=os.path.join(
        script_dir,
        "./tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "./tests/test_data/allen_mouse_2017_atlas//allen2017_colours.csv"
    ),
)
pnt.get_coordinates(object_cutoff=0, use_flat=False)
pnt.quantify_coordinates()
pnt.save_analysis("./test_result/2custom_atlas_hemi_test_24_03_2025")
