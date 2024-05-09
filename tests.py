import unittest
from utils import PlotData
import pandas as pd

class TestDeviationDataset(unittest.TestCase):
    def test_check_missing_values(self):
        d = pd.DataFrame({'a': [5.4, 2.8, None, 8.34, 0.1], 'b': [None, None, 7, 8, 5], 'c': [1, 4, 6, 7, 9]})
        p = PlotData()
        mv = p.check_missing_values(data=d)
        self.assertTrue((mv).equals(pd.Series({'a': 1, 'b': 2, 'c': 0})))

    def test_r2_score(self):
        d = pd.DataFrame({'y_true': [2, 4, 5, 1], 'y_pred': [2, 3, 5, 1]})
        p = PlotData()
        r2 = p.r2_score(y_true=d['y_true'].to_numpy(), y_pred=d['y_pred'].to_numpy())
        self.assertAlmostEqual(r2, 0.9, 2)

    def test_accuracy(self):
        d = pd.DataFrame({'y_true': [2, 4, 5, 1], 'y_pred': [2, 3, 5, 1]})
        p = PlotData()
        r2 = p.accuracy(y_true=d['y_true'].to_numpy(), y_pred=d['y_pred'].to_numpy())
        self.assertAlmostEqual(r2, 0.75, 2)


if __name__ == "__main__":
    unittest.main()