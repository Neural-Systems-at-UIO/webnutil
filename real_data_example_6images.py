import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neutil import Neutil

script_dir = os.path.dirname(os.path.abspath(__file__))

nt = Neutil(
    segmentation_folder=os.path.join(
        script_dir, "tests/test_real/segmentations"
    ),
    alignment_json=os.path.join(
        script_dir, "tests/test_real/aba_mouse_ccfv3_2017_25um_2025-04-11_07-50-43.json"
    ),
    colour=[0, 0, 0],
    atlas_path=os.path.join(
        script_dir,
        "tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    ),
    label_path=os.path.join(
        script_dir, "tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv"
    ),
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()
print(nt.get_region_summary().sort_values(by="object_count", ascending=False))
nt.save_analysis("./test_result/aba_mouse")
