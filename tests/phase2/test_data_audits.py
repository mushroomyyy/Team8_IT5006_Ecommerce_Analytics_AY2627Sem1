import unittest

import pandas as pd

from src import data_audits as audits
from src.feature_selection import candidate_feature_status, candidate_feature_table


class DataAuditTests(unittest.TestCase):
    def test_candidate_groups_cover_41_features(self):
        self.assertEqual(candidate_feature_table()['count'].sum(), 41)
        self.assertEqual(len(candidate_feature_status()), 41)

    def test_skewness_report_flags_only_skewed_candidates(self):
        rows = pd.DataFrame({'payment_value_sum': [1.0] * 50 + [1000.0],
                             'order_item_count': [1.0] * 50 + [30.0],
                             'approval_lag_hours': [1.0, 2.0, 3.0] * 17})
        report = audits.skewness_report(rows, list(rows.columns)).set_index('feature')
        self.assertTrue(report.loc['payment_value_sum', 'log1p_applied'])
        self.assertFalse(report.loc['order_item_count', 'log1p_applied'])
        self.assertFalse(report.loc['approval_lag_hours', 'log1p_applied'])

    def test_route_audit_counts_late_orders(self):
        rows = pd.DataFrame({'route_type': ['a', 'a', '3. Mixed'], 'seller_state_count': [1, 2, 2],
                             'y': [1, 0, 0]})
        table = audits.route_audit(rows, 'classification').set_index('group')
        self.assertEqual(table.loc['a', 'late_orders'], 1)
        self.assertEqual(table.loc['seller_state_count >= 2', 'orders'], 2)


if __name__ == '__main__':
    unittest.main()
