import unittest

import pandas as pd

from classification_eval import daily_capacity


class DailyCapacityTests(unittest.TestCase):
    def test_capacity_uses_every_scored_order_and_splits_ties(self):
        rows = pd.DataFrame({
            'approval_date': ['2018-06-01'] * 4 + ['2018-06-02'] * 2,
            'score': [0.9, 0.9, 0.2, 0.1, 0.8, 0.1],
            'actual_label': [1, pd.NA, 0, 0, 1, 0],
        })
        result = daily_capacity(rows, fraction=0.25)
        self.assertEqual(result['selected'], 2)
        self.assertEqual(result['known'], 5)
        self.assertAlmostEqual(result['known_selected'], 1.5)
        self.assertAlmostEqual(result['unknown_selected'], 0.5)
        self.assertAlmostEqual(result['coverage'], 0.75)

    def test_fraction_and_probability_are_checked(self):
        rows = pd.DataFrame({'approval_date': ['2018-06-01'],
                             'score': [1.2], 'actual_label': [1]})
        with self.assertRaises(ValueError):
            daily_capacity(rows)
        with self.assertRaises(ValueError):
            daily_capacity(rows.assign(score=0.5), fraction=0)


if __name__ == '__main__':
    unittest.main()
