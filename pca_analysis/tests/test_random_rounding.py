import unittest
import numpy as np
import math

class ProbRoundTests(unittest.TestCase):
 
    def setUp(self):
        np.random.seed(42)
 
    def test_prob_round_negative_number(self):
        x = np.array([-3.5])
        result = prob_round(x)
        self.assertEqual(result[0], -3)
 
    def test_prob__positive_number(self):
        x = np.array([4.9999999])
        result = prob_round(x)
        self.assertEqual(result[0], 5)

def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = np.random.random(x.shape) < x - x.astype(int)
    round_func = math.ceil if is_up else math.floor
    return sign * round_func(x)

if __name__ == "__main__":
    unittest.main()