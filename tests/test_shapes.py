import unittest
import numpy as np
from pathlib import Path

class TestShapeLoading(unittest.TestCase):
    def setUp(self):
        self.shape_file = Path("shapes/bunny2D.csv")
    
    def test_shape_file_exists(self):
        """Test if the shape file exists"""
        self.assertTrue(self.shape_file.exists())

    def test_shape_file_readable(self):
        """Test if the shape file can be read as CSV"""
        try:
            data = np.loadtxt(self.shape_file, delimiter=",")
            self.assertIsNotNone(data)
            self.assertTrue(len(data.shape) == 2)  # Should be 2D array
            self.assertTrue(data.shape[1] == 2)    # Should have 2 columns (x,y)
        except Exception as e:
            self.fail(f"Failed to load shape file: {e}")

if __name__ == '__main__':
    unittest.main()
