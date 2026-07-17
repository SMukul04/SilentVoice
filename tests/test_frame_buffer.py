import unittest
import numpy as np

from backend.sign_recognition.hand_features import HandFeatures
from backend.sign_recognition.frame_buffer import FrameBuffer


class TestFrameBuffer(unittest.TestCase):
    """Unit tests for the FrameBuffer class."""

    def setUp(self) -> None:
        """Sets up a FrameBuffer instance for testing."""
        self.buffer_size = 5
        self.frame_buffer = FrameBuffer(buffer_size=self.buffer_size)

        # Create valid mock hand features
        self.lms_valid = np.zeros(63, dtype=np.float32)
        self.hf_valid = HandFeatures(
            landmarks=self.lms_valid,
            handedness="Right",
            confidence=0.95
        )

    def test_init_invalid_size(self) -> None:
        """Tests that initializing FrameBuffer with an invalid size raises ValueError."""
        with self.assertRaises(ValueError):
            FrameBuffer(buffer_size=0)
        with self.assertRaises(ValueError):
            FrameBuffer(buffer_size=-5)

    def test_add_valid_features(self) -> None:
        """Tests adding valid hand features to the buffer."""
        self.assertEqual(len(self.frame_buffer), 0)
        self.assertFalse(self.frame_buffer.is_full())

        self.frame_buffer.add(self.hf_valid)
        self.assertEqual(len(self.frame_buffer), 1)

    def test_add_invalid_features_raises_exception(self) -> None:
        """Tests that adding invalid features raises correct exceptions."""
        # Case 1: Wrong type
        with self.assertRaises(TypeError):
            self.frame_buffer.add("not_hand_features")  # type: ignore

        # Case 2: HandFeatures with invalid landmark shape (bypassed via post-init modification)
        hf_invalid = HandFeatures(
            landmarks=np.zeros(63, dtype=np.float32),
            handedness="Right",
            confidence=0.9
        )
        hf_invalid.landmarks = np.zeros(60, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.frame_buffer.add(hf_invalid)

    def test_sliding_window_discard(self) -> None:
        """Tests that buffer maxlen behavior automatically drops oldest items."""
        # Fill buffer with self.buffer_size (5) distinct elements
        for i in range(self.buffer_size):
            hf = HandFeatures(
                landmarks=np.ones(63, dtype=np.float32) * float(i),
                handedness="Right",
                confidence=0.9
            )
            self.frame_buffer.add(hf)

        self.assertEqual(len(self.frame_buffer), 5)
        self.assertTrue(self.frame_buffer.is_full())

        # Oldest element value in buffer is 0.0, newest is 4.0
        seq_before = self.frame_buffer.get_sequence()
        np.testing.assert_array_equal(seq_before[0], np.ones(63, dtype=np.float32) * 0.0)
        np.testing.assert_array_equal(seq_before[-1], np.ones(63, dtype=np.float32) * 4.0)

        # Add a 6th element (value 5.0). Buffer size should remain 5.
        hf_new = HandFeatures(
            landmarks=np.ones(63, dtype=np.float32) * 5.0,
            handedness="Right",
            confidence=0.9
        )
        self.frame_buffer.add(hf_new)

        self.assertEqual(len(self.frame_buffer), 5)
        self.assertTrue(self.frame_buffer.is_full())

        # Oldest element (0.0) should have been dropped. New oldest is 1.0, newest is 5.0.
        seq_after = self.frame_buffer.get_sequence()
        np.testing.assert_array_equal(seq_after[0], np.ones(63, dtype=np.float32) * 1.0)
        np.testing.assert_array_equal(seq_after[-1], np.ones(63, dtype=np.float32) * 5.0)

    def test_clear_buffer(self) -> None:
        """Tests clearing the buffer contents."""
        self.frame_buffer.add(self.hf_valid)
        self.assertEqual(len(self.frame_buffer), 1)

        self.frame_buffer.clear()
        self.assertEqual(len(self.frame_buffer), 0)
        np.testing.assert_array_equal(self.frame_buffer.get_sequence(), np.empty((0, 63), dtype=np.float32))

    def test_get_sequence_empty(self) -> None:
        """Tests get_sequence when buffer is empty."""
        seq = self.frame_buffer.get_sequence()
        self.assertEqual(seq.shape, (0, 63))


if __name__ == '__main__':
    unittest.main()
