import unittest

import pandas as pd

from src import model_labels as ml


class ModelLabelTests(unittest.TestCase):
    def test_every_code_has_short_label_and_description(self):
        for code in ml.CLASSIFICATION_CODES + ml.REGRESSION_CODES:
            self.assertTrue(ml.short_label(code) and ml.description(code), code)
        self.assertEqual(len(ml.MODEL_LABELS), 13)

    def test_display_name(self):
        self.assertEqual(ml.display_name('C2'), 'C2 · Logistic (CV-stepwise)')
        with self.assertRaises(KeyError):
            ml.display_name('C9')

    def test_add_model_columns_puts_code_and_model_first(self):
        table = pd.DataFrame({'model': ['C0', 'C3d'], 'AP': [0.1, 0.2]})
        out = ml.add_model_columns(table)
        self.assertEqual(list(out.columns), ['Code', 'Model', 'AP'])
        self.assertEqual(out['Model'].tolist(), ['No-skill baseline', 'Random Forest (default)'])
        self.assertEqual(list(table.columns), ['model', 'AP'])  # input untouched

    def test_add_model_columns_custom_code_column(self):
        out = ml.add_model_columns(pd.DataFrame({'Model': ['R1'], 'RMSE': [7.0]}), code_col='Model')
        self.assertEqual(list(out.columns), ['Code', 'Model', 'RMSE'])
        self.assertEqual(out.loc[0, 'Model'], 'OLS (all)')

    def test_label_model_columns_only_renames_codes(self):
        out = ml.label_model_columns(pd.DataFrame({'Decile': [1], 'C1': [0.5]}))
        self.assertEqual(list(out.columns), ['Decile', 'C1 · Logistic (all)'])

    def test_model_key_table_uses_given_feature_counts(self):
        key = ml.model_key_table(['C0', 'C2'], {'C2': 3})
        self.assertEqual(list(key.columns), ['Code', 'Model', 'Description', 'Features'])
        self.assertEqual(key['Features'].tolist(), ['-', 3])

    def test_promise_description_matches_baseline(self):
        from src.regression_models import PromiseBaseline
        pred = PromiseBaseline().fit(pd.DataFrame({'promised_lead_days': [10]}), [0]).predict(
            pd.DataFrame({'promised_lead_days': [10, 60]}))
        self.assertEqual(pred[0], 0)
        self.assertLess(pred[1], 0)  # promises beyond the waiting period are capped
        self.assertIn('0 days', ml.description('R0b'))


if __name__ == '__main__':
    unittest.main()
