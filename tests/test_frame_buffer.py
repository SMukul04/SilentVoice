import unittest
import numpy as np

from backend.sign_recognition.frame_buffer import FrameBuffer


class TestFrameBuffer(unittest.TestCase):
    """Unit tests for the FrameBuffer class."""

    def setUp(self) -> None:
        """Sets up a FrameBuffer instance for testing."""
        self.buffer_size = 30
        self.frame_buffer = FrameBuffer(buffer_size=self.buffer_size)

    def test_init_invalid_size(self) -> None:
        """Tests that initializing FrameBuffer with an invalid size raises ValueError."""
        with self.assertRaises(ValueError):
            FrameBuffer(buffer_size=0)
        with self.assertRaises(ValueError):
            FrameBuffer(buffer_size=-5)

    def test_empty_buffer(self) -> None:
        """Tests size and is_full on an empty buffer."""
        self.assertEqual(self.frame_buffer.size(), 0)
        self.assertFalse(self.frame_buffer.is_full())
        self.assertEqual(self.frame_buffer.get_sequence().shape, (0, 126))

    def test_append_one_frame(self) -> None:
        """Tests appending a single frame."""
        frame = np.zeros(126, dtype=np.float32)
        self.frame_buffer.append(frame)
        self.assertEqual(self.frame_buffer.size(), 1)
        self.assertFalse(self.frame_buffer.is_full())

    def test_append_15_frames(self) -> None:
        """Tests appending 15 frames."""
        for _ in range(15):
            frame = np.zeros(126, dtype=np.float32)
            self.frame_buffer.append(frame)
        self.assertEqual(self.frame_buffer.size(), 15)
        self.assertFalse(self.frame_buffer.is_full())

    def test_append_exactly_30_frames(self) -> None:
        """Tests appending exactly 30 frames."""
        for i in range(30):
            frame = np.ones(126, dtype=np.float32) * float(i)
            self.frame_buffer.append(frame)
        self.assertEqual(self.frame_buffer.size(), 30)
        self.assertTrue(self.frame_buffer.is_full())
        
        sequence = self.frame_buffer.get_sequence()
        self.assertEqual(sequence.shape, (30, 126))
        # Verify chronological order
        np.testing.assert_array_equal(sequence[0], np.zeros(126, dtype=np.float32))
        np.testing.assert_array_equal(sequence[-1], np.ones(126, dtype=np.float32) * 29.0)

    def test_append_40_frames_sliding_window(self) -> None:
        """Tests appending 40 frames to verify the sliding-window behavior."""
        for i in range(40):
            frame = np.ones(126, dtype=np.float32) * float(i)
            self.frame_buffer.append(frame)

        # Oldest 10 frames are discarded, latest 30 frames remain
        self.assertEqual(self.frame_buffer.size(), 30)
        self.assertTrue(self.frame_buffer.is_full())

        sequence = self.frame_buffer.get_sequence()
        self.assertEqual(sequence.shape, (30, 126))
        # Oldest remaining should be 10.0, newest should be 39.0
        np.testing.assert_array_equal(sequence[0], np.ones(126, dtype=np.float32) * 10.0)
        np.testing.assert_array_equal(sequence[-1], np.ones(126, dtype=np.float32) * 39.0)

    def test_invalid_inputs(self) -> None:
        """Tests that appending invalid inputs raises correct exceptions."""
        # Case 1: Python list instead of NumPy array -> TypeError
        with self.assertRaises(TypeError):
            self.frame_buffer.append([0.0] * 126)  # type: ignore

        # Case 2: Shape (63,) -> ValueError
        with self.assertRaises(ValueError):
            self.frame_buffer.append(np.zeros(63, dtype=np.float32))

        # Case 3: Shape (126, 1) -> ValueError
        with self.assertRaises(ValueError):
            self.frame_buffer.append(np.zeros((126, 1), dtype=np.float32))

    def test_clear_buffer(self) -> None:
        """Tests clearing the buffer contents."""
        frame = np.zeros(126, dtype=np.float32)
        self.frame_buffer.append(frame)
        self.assertEqual(self.frame_buffer.size(), 1)

        self.frame_buffer.clear()
        self.assertEqual(self.frame_buffer.size(), 0)
        self.assertEqual(self.frame_buffer.get_sequence().shape, (0, 126))


if __name__ == '__main__':
    unittest.main()
