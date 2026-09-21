import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from classification_experiments import enriched_features, EXTRA_NUM, EXTRA_CAT, model_input, make_model
from classification_significance import weighted_ap, holm, compare
from model_workbench import load_order_tables
from classification_time_distance import haversine_km, geographic_lookup, purchase_distance_features
from classification_ensembles import equal_weight_blend
from classification_metric_report import summarise


class ExperimentTests(unittest.TestCase):
    def test_metric_report_uses_saved_decisions(self):
        pred = pd.DataFrame(dict(actual=[0, 1, 0, 1], probability=[.1, .9, .2, .8],
            predicted=[0, 0, 0, 0], fold=[1]*4, approval_date=['2018-01-01']*4))
        result = summarise(pred)
        self.assertEqual(result['mean_fold_ap'], 1.)
        self.assertEqual(result['forward_recall'], 0.)
        self.assertEqual(result['confusion_matrix'], dict(tn=2, fp=0, fn=2, tp=0))
        self.assertAlmostEqual(result['brier_score'], .025)
        self.assertAlmostEqual(result['probability_rmse']**2, .025)
        self.assertAlmostEqual(result['capacity']['precision_at_10pct'], 1.)
        self.assertAlmostEqual(result['capacity']['coverage_at_10pct'], .5)

    def test_ensemble_alignment_and_label_independence(self):
        base = pd.DataFrame(dict(order_id=['a', 'b'], approval_date=['2018-01-01']*2,
            label_available_at=['2018-01-20']*2, actual=[0, 1], fold=[1, 1],
            sample=['cross_validation']*2, probability=[.1, .6], model=['one']*2))
        second = base.copy()
        second['model'], second['probability'] = 'two', [.3, .8]
        frame = pd.concat([base, second.iloc[::-1]], ignore_index=True)
        result = equal_weight_blend(frame, ['one', 'two'], 'blend')
        np.testing.assert_allclose(result.probability, [.2, .7])
        frame['actual'] = 1 - frame.actual
        np.testing.assert_array_equal(result.probability,
            equal_weight_blend(frame, ['one', 'two'], 'blend').probability)
        frame.loc[frame.model.eq('two'), 'fold'] = 2
        with self.assertRaises(ValueError):
            equal_weight_blend(frame, ['one', 'two'], 'blend')

    def test_geographic_reference(self):
        self.assertAlmostEqual(float(haversine_km(0, 0, 0, 0)), 0.)
        self.assertAlmostEqual(float(haversine_km(0, 0, 0, 1)), 111.195, places=2)
        self.assertTrue(np.isnan(haversine_km(np.nan, 0, 0, 1)))
        geo = pd.DataFrame({'geolocation_zip_code_prefix': [1, 1, 1, 2],
            'geolocation_lat': [-20., -22., 90., -15.],
            'geolocation_lng': [-40., -42., 150., -45.]})
        lookup = geographic_lookup(geo)
        self.assertEqual(lookup.loc[1, 'geolocation_lat'], -21.)
        self.assertEqual(lookup.loc[1, 'geolocation_lng'], -41.)

    def test_weighted_ap_matches_sklearn_with_ties(self):
        y = np.array([0, 1, 1, 0, 1])
        scores = np.array([.5, .5, .2, .1, .9])
        days = np.array([0, 0, 1, 1, 2])
        weights = np.array([2., 0., 3.])
        self.assertAlmostEqual(weighted_ap(y, scores, days)(weights),
            average_precision_score(y, scores, sample_weight=weights[days]))

    def test_identical_predictions_null(self):
        rows = []
        for name in ['base_logistic_regression', 'base_catboost']:
            for i in range(20):
                rows.append(dict(model=name, order_id=str(i), fold=1, actual=i % 2,
                    approval_date=pd.Timestamp('2018-01-01') + pd.Timedelta(days=i), probability=i / 20))
        result = compare(pd.DataFrame(rows), repeats=19)
        self.assertEqual(result['comparisons'][0]['p_holm'], 1.)
        self.assertEqual(result['comparisons'][0]['ci95_percentile'], [0., 0.])
        self.assertEqual(holm([.01, .04, .03]), [.03, .06, .06])

    def test_enrichment_is_outcome_independent(self):
        tables = load_order_tables(Path(__file__).resolve().parents[1] / 'streamlit_release/data/olist_csv.zip')
        first = enriched_features(tables, '2018-01-01')
        tables['orders'] = tables['orders'].drop(columns=['order_status',
            'order_delivered_customer_date', 'order_delivered_carrier_date'])
        second = enriched_features(tables, '2018-01-01')
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.order_id.is_unique)
        self.assertTrue(set(EXTRA_NUM + EXTRA_CAT).issubset(first.columns))
        x = model_input(first.head(100), 'catboost', True)
        model = make_model('catboost', True, 1.)
        model.set_params(iterations=2)
        model.fit(x, np.arange(len(x)) % 2)
        self.assertEqual(model.predict_proba(x).shape, (100, 2))
        # Sparse reference deliberately exercises missing-coordinate handling.
        prefix = tables['customers'].customer_zip_code_prefix.iloc[0]
        geo = pd.DataFrame({'geolocation_zip_code_prefix': [prefix],
            'geolocation_lat': [-20.], 'geolocation_lng': [-40.]})
        extended = purchase_distance_features(tables, geo, '2018-01-01')
        self.assertTrue(extended.order_id.is_unique)
        self.assertTrue(extended.seller_distance_missing_fraction.between(0, 1).all())
        source = tables['orders'].set_index('order_id').loc[extended.order_id]
        np.testing.assert_array_equal(extended.purchase_weekday, source.order_purchase_timestamp.dt.dayofweek)
        self.assertTrue(extended.approval_date.lt(pd.Timestamp('2018-01-01')).all())


if __name__ == '__main__':
    unittest.main()
