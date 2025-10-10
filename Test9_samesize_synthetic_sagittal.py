import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from nutil import Nutil

nt = Nutil(
    segmentation_folder="tests/test_data/Test9_data/segmentations/",
    alignment_json="tests/test_data/Test9_data/Test9_synthetic_sagittal_nonlin.json",
    colour=[0, 0, 0],
    atlas_path="tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    label_path="tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv",
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()

nt.save_analysis("./test_result/Test9_12_09_25")
