import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from nutil import Nutil

nt = Nutil(
    segmentation_folder="tests/test_data/Test10-17_Arda_synthetic_data/segmentations-double/",
    alignment_json="tests/test_data/Test10-17_Arda_synthetic_data/Test12-13_Arda_AMBA_horizontal_nonlin.json",
    colour=[0, 0, 0],
    atlas_path="tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    label_path="tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv",
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()
print(nt.get_region_summary().sort_values(by="object_count", ascending=False).head(20)[["idx", "name", "object_count"]])
nt.save_analysis("./test_result/Test13_Arda_AMBA_horizontal_double")

