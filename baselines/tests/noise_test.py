import unittest
from baselines import NormalNoise
import numpy as np

class NoiseTest(unittest.TestCase):
    def test_apply_0_noise(self):
        n = NormalNoise(0)
        input = 2.5*np.ones(shape=(1,),dtype=np.float32)
        self.assertAlmostEqual(n.apply(input),2.5,delta=1e-8)




if __name__ == '__main__':
    unittest.main()
