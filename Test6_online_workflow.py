import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from nutil import Nutil

nt = Nutil(
    segmentation_folder="tests/test_data/Test6_workflow/segmentation/",
    alignment_json="tests/test_data/Test6_workflow/aba_mouse_ccfv3_2017_25um_2025-09-03_11-39-23.waln",
    colour=[0, 0, 255],
    atlas_path="tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
    label_path="tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv",
)
nt.get_coordinates(object_cutoff=0, use_flat=False)
nt.quantify_coordinates()
print(nt.get_region_summary().sort_values(by="object_count", ascending=False).head(20)[["idx", "name", "object_count"]])
nt.save_analysis("./test_result/Test6_online_workflow_outputs")

