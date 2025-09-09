import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import sys

from nutil.core.nutil import Nutil

sys.path.append(str(Path(__file__).parent))


class TestNutilResultsSynthetic:
    """Test functional analysis outputs with synthetic datasets."""
    
    @pytest.fixture
    def synthetic_data_paths(self):
        """Provide paths to synthetic test data double size"""
        base_path = Path("tests/test_data/synthetic_data_doublesize_nonlinear")
        return {
            "segmentation_folder": str(base_path / "segmentations"),
            "alignment_json": str(base_path / "alignment2x.json"),
            "atlas_path": "tests/test_data/allen_mouse_2017_atlas/annotation_25_reoriented_2017.nrrd",
            "label_path": "tests/test_data/allen_mouse_2017_atlas/allen2017_colours.csv",
            "expected_report": base_path / "flat_files" / "report.tsv" 
        }
    
    @pytest.fixture
    def expected_ground_truth(self, synthetic_data_paths):
        """Load ground truth data from synthetic dataset."""
        report_path = synthetic_data_paths["expected_report"]
        if report_path.exists():
            ground_truth = pd.read_csv(report_path, sep='\t')
            return ground_truth
        # Ground Truth is provided by analysis in Nutil
        return None

    def test_synthetic_data_object_detection_basic(self, synthetic_data_paths):
        """Test basic object detection with synthetic data."""
        if not Path(synthetic_data_paths["segmentation_folder"]).exists():
            pytest.skip("Synthetic test data not available")
            
        nutil_object = Nutil(
            segmentation_folder=synthetic_data_paths["segmentation_folder"],
            alignment_json=synthetic_data_paths["alignment_json"],
            colour=[0, 0, 0],  # Black pixels (objects)
            atlas_path=synthetic_data_paths["atlas_path"],
            label_path=synthetic_data_paths["label_path"],
        )
        
        nutil_object.get_coordinates(object_cutoff=0, use_flat=False)
        nutil_object.quantify_coordinates()
        
        results = nutil_object.get_region_summary().sort_values(by="object_count", ascending=False)
        
        # Basic sanity checks
        assert not results.empty, "Region summary should not be empty"
        assert results["object_count"].sum() > 0, "Should detect some objects"
        assert len(results) > 0, "Should have regions with objects"
        # TODO add region names and ids checks
        
        expected_total = 30 # With no cutoff should total 30 objects
        actual_total = results["object_count"].sum()
        
        # Allow some tolerance for the ground truths from Nutil and webnutil
        tolerance = 0.1  # 10% tolerance - 
        assert abs(actual_total - expected_total) / expected_total <= tolerance, \
            f"Object count {actual_total} deviates too much from expected {expected_total}"

    def test_synthetic_data_section_wise_analysis(self, synthetic_data_paths):
        """Test section-wise analysis for consistency."""
        if not Path(synthetic_data_paths["segmentation_folder"]).exists():
            pytest.skip("Synthetic test data not available")
            
        nutil_object = Nutil(
            segmentation_folder=synthetic_data_paths["segmentation_folder"],
            alignment_json=synthetic_data_paths["alignment_json"],
            colour=[0, 0, 0],
            atlas_path=synthetic_data_paths["atlas_path"],
            label_path=synthetic_data_paths["label_path"],
        )
        
        nutil_object.get_coordinates(object_cutoff=0, use_flat=False)
        nutil_object.quantify_coordinates()
        
        # Access per-section data
        if hasattr(nutil_object, 'per_section_df') and nutil_object.per_section_df is not None:
            per_section = nutil_object.per_section_df
            
            if isinstance(per_section, pd.DataFrame):
                # Verify we have data for 5 sections (test_s001 to test_s005)
                assert len(per_section) >= 5, "Should have data for at least 5 sections"
                # TODO for other sets implement folder check
                
                # Check each section has reasonable object counts
                for _, section in per_section.iterrows():
                    assert section.get('object_count', 0) >= 0, "Object count should be non-negative"
            elif isinstance(per_section, list):
                assert len(per_section) >= 5, "Should have data for at least 5 sections"
                for section_df in per_section:
                    if isinstance(section_df, pd.DataFrame) and not section_df.empty:
                        assert 'object_count' in section_df.columns or len(section_df) >= 0

    def test_synthetic_data_parameter_variations(self, synthetic_data_paths):
        """Test different parameter combinations for robustness."""
        if not Path(synthetic_data_paths["segmentation_folder"]).exists():
            pytest.skip("Synthetic test data not available")
            
        base_params = {
            "segmentation_folder": synthetic_data_paths["segmentation_folder"],
            "alignment_json": synthetic_data_paths["alignment_json"],
            "colour": [0, 0, 0],
            "atlas_path": synthetic_data_paths["atlas_path"],
            "label_path": synthetic_data_paths["label_path"],
        }
        
        # Test different object cutoffs
        # TODO WIP
        cutoff_results = {}
        for cutoff in [0, 5, 10]:
            nutil_obj = Nutil(**base_params)
            nutil_obj.get_coordinates(object_cutoff=cutoff, use_flat=False)
            nutil_obj.quantify_coordinates()
            
            results = nutil_obj.get_region_summary()
            cutoff_results[cutoff] = results["object_count"].sum()
        
        # Higher cutoffs should detect fewer or equal objects
        assert cutoff_results[0] >= cutoff_results[5] >= cutoff_results[10], \
            "Higher cutoffs should filter out smaller objects"

    def test_synthetic_data_coordinate_consistency(self, synthetic_data_paths):
        """Test coordinate extraction consistency."""
        if not Path(synthetic_data_paths["segmentation_folder"]).exists():
            pytest.skip("Synthetic test data not available")
            
        nutil_object = Nutil(
            segmentation_folder=synthetic_data_paths["segmentation_folder"],
            alignment_json=synthetic_data_paths["alignment_json"],
            colour=[0, 0, 0],
            atlas_path=synthetic_data_paths["atlas_path"],
            label_path=synthetic_data_paths["label_path"],
        )
        
        nutil_object.get_coordinates(object_cutoff=0, use_flat=False)
        
        # Verify coordinate data structures exist and are reasonable
        # Mid analysis stage checks
        assert hasattr(nutil_object, 'pixel_points'), "Should have pixel points"
        assert hasattr(nutil_object, 'centroids'), "Should have centroids"
        assert hasattr(nutil_object, 'points_len'), "Should have points length per section"
        assert hasattr(nutil_object, 'centroids_len'), "Should have centroids length per section"
        
        # Basic coordinate validation
        if nutil_object.pixel_points is not None and len(nutil_object.pixel_points) > 0:
            # Check coordinate ranges are reasonable (atlas space)
            coords = np.array(nutil_object.pixel_points)
            assert coords.shape[1] == 3, "Coordinates should be 3D"
            assert np.all(coords >= 0), "Atlas coordinates should be non-negative"

    def test_synthetic_data_reproducibility(self, synthetic_data_paths):
        """Test that analysis is reproducible with same parameters."""
        if not Path(synthetic_data_paths["segmentation_folder"]).exists():
            pytest.skip("Synthetic test data not available")
            
        params = {
            "segmentation_folder": synthetic_data_paths["segmentation_folder"],
            "alignment_json": synthetic_data_paths["alignment_json"],
            "colour": [0, 0, 0],
            "atlas_path": synthetic_data_paths["atlas_path"],
            "label_path": synthetic_data_paths["label_path"],
        }
        
        # Run analysis twice
        results1 = self._run_full_analysis(params)
        results2 = self._run_full_analysis(params)
        
        # Results should be identical
        # Rudimentary check
        pd.testing.assert_frame_equal(
            results1.sort_index(), 
            results2.sort_index(),
            check_dtype=False,
            rtol=1e-10
        )
    
    def _run_full_analysis(self, params):
        """Helper method to run complete analysis."""
        nutil_obj = Nutil(**params)
        nutil_obj.get_coordinates(object_cutoff=0, use_flat=False)
        nutil_obj.quantify_coordinates()
        return nutil_obj.get_region_summary().sort_values(by="object_count", ascending=False).head(25)