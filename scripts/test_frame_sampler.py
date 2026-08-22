"""Test script for the FrameSampler module."""

from __future__ import annotations

from pathlib import Path
import sys

from backend.dataset.frame_sampler import FrameSampler


def run_tests() -> None:
    """Runs verification tests for the FrameSampler."""
    results = {}

    # TEST 1 — Long sequence
    try:
        # Create 69 fake Path objects representing sequential frames
        frame_paths_69 = [Path(f"frame_{i:04d}.jpg") for i in range(1, 70)]
        sampler_32 = FrameSampler(sequence_length=32)
        output_32 = sampler_32.sample(frame_paths_69)

        # Verify output length is exactly 32
        assert len(output_32) == 32, f"Length is {len(output_32)} instead of 32"
        # Verify first frame is preserved
        assert output_32[0] == frame_paths_69[0], "First frame not preserved"
        # Verify last frame is preserved
        assert output_32[-1] == frame_paths_69[-1], "Last frame not preserved"
        # Verify frames remain chronologically ordered
        for i in range(len(output_32) - 1):
            assert output_32[i] < output_32[i + 1], f"Order not preserved at index {i}"
        # Verify output is deterministic
        output_32_second = sampler_32.sample(frame_paths_69)
        assert output_32 == output_32_second, "Sampling is not deterministic"

        results["Test 1: Long sequence"] = "PASSED"
    except Exception as e:
        results["Test 1: Long sequence"] = f"FAILED: {e}"

    # TEST 2 — Exact sequence
    try:
        # Create exactly 32 frame paths
        frame_paths_32 = [Path(f"frame_{i:04d}.jpg") for i in range(1, 33)]
        output_exact = sampler_32.sample(frame_paths_32)

        # Verify returned sequence is unchanged
        assert len(output_exact) == 32, f"Length is {len(output_exact)} instead of 32"
        assert output_exact == frame_paths_32, "Exact sequence was modified"
        results["Test 2: Exact sequence"] = "PASSED"
    except Exception as e:
        results["Test 2: Exact sequence"] = f"FAILED: {e}"

    # TEST 3 — Short sequence
    try:
        # Create 20 frame paths
        frame_paths_20 = [Path(f"frame_{i:04d}.jpg") for i in range(1, 21)]
        output_padded = sampler_32.sample(frame_paths_20)

        # Verify output length is 32
        assert len(output_padded) == 32, f"Length is {len(output_padded)} instead of 32"
        # Verify first 20 frames remain unchanged
        assert list(output_padded[:20]) == frame_paths_20, "First 20 frames were modified"
        # Verify remaining frames are repetitions of the final frame
        for frame in output_padded[20:]:
            assert frame == frame_paths_20[-1], "Padded frame is not the final frame"

        results["Test 3: Short sequence"] = "PASSED"
    except Exception as e:
        results["Test 3: Short sequence"] = f"FAILED: {e}"

    # TEST 4 — Very short sequence
    try:
        # Create a sequence containing only 1 frame
        frame_paths_1 = [Path("frame_0001.jpg")]
        output_single = sampler_32.sample(frame_paths_1)

        # Verify that all 32 returned paths refer to that same frame
        assert len(output_single) == 32, f"Length is {len(output_single)} instead of 32"
        for frame in output_single:
            assert frame == frame_paths_1[0], "Padded frame does not refer to the single frame"

        results["Test 4: Single frame sequence"] = "PASSED"
    except Exception as e:
        results["Test 4: Single frame sequence"] = f"FAILED: {e}"

    # TEST 5 — Empty sequence
    try:
        # Pass an empty sequence
        sampler_32.sample([])
        results["Test 5: Empty sequence"] = "FAILED: Did not raise an exception"
    except ValueError:
        results["Test 5: Empty sequence"] = "PASSED"
    except Exception as e:
        results["Test 5: Empty sequence"] = f"FAILED: Raised unexpected exception: {e}"

    # TEST 6 — Invalid sequence length
    try:
        # Attempt to create FrameSampler(sequence_length=0)
        try:
            FrameSampler(sequence_length=0)
            zero_ok = False
        except ValueError:
            zero_ok = True

        # Attempt to create negative sequence length
        try:
            FrameSampler(sequence_length=-5)
            neg_ok = False
        except ValueError:
            neg_ok = True

        # Attempt to create with non-integer type
        try:
            FrameSampler(sequence_length=3.5)  # type: ignore
            float_ok = False
        except TypeError:
            float_ok = True

        try:
            FrameSampler(sequence_length=True)  # type: ignore
            bool_ok = False
        except TypeError:
            bool_ok = True

        assert zero_ok and neg_ok and float_ok and bool_ok, "Invalid sequence length not rejected correctly"
        results["Test 6: Invalid sequence length"] = "PASSED"
    except Exception as e:
        results["Test 6: Invalid sequence length"] = f"FAILED: {e}"

    # Print a clean report of standard tests
    print("===================================")
    print("FRAME SAMPLER TEST")
    print("===================================")
    print()
    for name, status in results.items():
        print(name)
        print(status)
        print()

    # REAL DATASET INTEGRATION
    dataset_root = Path("datasets/processed")
    real_sample_info = None

    if dataset_root.exists() and dataset_root.is_dir():
        try:
            from backend.dataset.indexer import DatasetIndexer
            indexer = DatasetIndexer(dataset_root)
            indexer.build_index()

            samples = indexer.get_samples()
            if samples:
                real_sample = samples[0]
                sampled_paths = sampler_32.sample(real_sample.frame_paths)
                real_sample_info = {
                    "sample_id": real_sample.sample_id,
                    "original_frames": real_sample.num_frames,
                    "sampled_frames": len(sampled_paths),
                    "first_frame": sampled_paths[0],
                    "last_frame": sampled_paths[-1],
                }
        except Exception as e:
            print(f"Error during real dataset integration test: {e}")

    if real_sample_info:
        print("===================================")
        print("REAL DATASET INTEGRATION")
        print("===================================")
        print()
        print(f"Sample ID: {real_sample_info['sample_id']}")
        print(f"Original Frames: {real_sample_info['original_frames']}")
        print(f"Sampled Frames: {real_sample_info['sampled_frames']}")
        print(f"First Frame: {real_sample_info['first_frame']}")
        print(f"Last Frame: {real_sample_info['last_frame']}")
        print()

    # Exit with non-zero code if any standard test failed
    any_failed = any("FAILED" in status for status in results.values())
    if any_failed:
        print("Some FrameSampler tests FAILED!")
        sys.exit(1)
    else:
        print("All FrameSampler tests completed successfully!")


if __name__ == "__main__":
    run_tests()
