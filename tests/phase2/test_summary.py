import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
from src import model_labels as ml
from src import summary as sm
from src.report_tables import export_report_tables


def toy_root(root):
    """Minimal exports for both tasks, in the layout notebooks 02 and 03 write."""
    for spec in sm.TASKS.values():
        out = Path(root) / spec.name
        models = [m for _, m in spec.rungs]
        rows = [{'model': m, 'split': s, spec.metric: 0.1 * (i + 1)}
                for i, m in enumerate(models) for s in ('Train', 'Validation', 'June', 'July', 'August')]
        out.mkdir(parents=True)
        pd.DataFrame(rows).to_csv(out / 'model_results_long.csv', index=False)
        pd.DataFrame([{'model': m, 'fold': k, spec.metric: 0.1 * (i + 1) + 0.01 * k}
                      for i, m in enumerate(models) for k in (1, 2, 3)]).to_csv(out / 'cv_fold_scores.csv', index=False)
        gaps = pd.DataFrame({'model': models, spec.metric: [0.01] * len(models)})
        drift = pd.DataFrame([{'model': m, 'metric': spec.metric, 'June': 1.0, 'July': 2.0, 'August': 3.0,
                               'change_june_to_last': 2.0} for m in models])
        export_report_tables([{'frame': ml.add_model_columns(gaps), 'title': 'toy overfitting gap'}], out,
                             f'{spec.name}_overfitting_gaps')
        export_report_tables([{'frame': ml.add_model_columns(drift), 'title': 'toy drift'}], out, f'{spec.name}_drift')
        (out / 'run_metadata.json').write_text(json.dumps({
            'versions': {'python': '3.13', 'numpy': spec.name[:1]},
            'features': {'numeric': ['a', 'b'], 'categorical': ['c']},
            'selected_variables': {spec.linear_models[0]: ['a', 'b', 'c'], spec.linear_models[1]: ['a']}}))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        toy_root(self.tmp.name)
        self.tasks = [sm.load_task(n, self.tmp.name) for n in sm.TASKS]

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_lookup_by_title(self):
        table = self.tasks[0].table('classification_drift', 'drift')
        self.assertEqual(len(table), 5)
        with self.assertRaises(KeyError):
            self.tasks[0].table('classification_drift', 'missing')

    def test_ladder_uses_cv_mean_and_display_scale(self):
        frame = sm.ladder_frame(self.tasks[0])
        c1 = frame[(frame['model'] == 'C1') & (frame['split'] == 'CV')].iloc[0]
        self.assertAlmostEqual(c1['value'], 100 * (0.2 + 0.02))
        self.assertEqual(len(frame), 15)

    def test_model_key_reads_feature_counts_from_metadata(self):
        text = '\n'.join(sm.model_key_markdown(self.tasks))
        self.assertIn('| C2 | Logistic (CV-stepwise) |', text)
        self.assertIn('Logistic (CV-stepwise) | Plain logistic, features chosen by CV forward stepwise with the 1-SE rule | 1 |', text)
        self.assertIn('| R3d | Random Forest (default) | Random Forest, raw 24 features, scikit-learn default hyperparameters | 3 |', text)

    def test_figures_are_written(self):
        for plot in (sm.plot_ladder, sm.plot_overfitting_gaps, sm.plot_drift_summary):
            path = Path(self.tmp.name) / f'{plot.__name__}.png'
            matplotlib.pyplot.close(plot(self.tasks, str(path)))
            self.assertTrue(path.exists())

    def test_top_effects_ranks_by_size_and_skips_blanks(self):
        table = pd.DataFrame({'Term': ['a', 'b', 'c'], 'Basis': ['per 1 unit'] * 3, 'OR per unit': ['1.1', '3.0', ''],
                              '95% CI per unit': ['x', 'y', ''], 'OR per 1 SD': ['', '', ''],
                              '95% CI per 1 SD': ['', '', ''], 'p-value': ['0.1', '<0.001', '']})
        rows = sm.top_effects(table, sm.TASKS['classification'])
        self.assertEqual([r[0] for r in rows], ['b', 'a'])

    def test_merge_metadata_flags_version_conflicts(self):
        merged = sm.merge_metadata(self.tasks)
        self.assertEqual(merged['package_version_conflicts'], ['numpy'])
        self.assertEqual(set(merged['tasks']), {'classification', 'regression'})

    def test_transformations_section_states_the_prespecified_design(self):
        task = self.tasks[0]
        pd.DataFrame({'feature': ['a', 'b'], 'skew_raw': [1.7, 11.2]}).to_csv(
            task.directory / 'skewness_train.csv', index=False)
        task.metadata['linear_transforms'] = {'log_cols': ['a', 'b'], 'cyclic': True, 'merge_mixed_route': True,
                                              'squared_cols': [], 'rule': 'pre-specified'}
        text = '\n'.join(sm._transforms(task))
        self.assertIn('`a`, `b`', text)
        self.assertIn('from 1.7 to 11.2', text)
        self.assertIn('No squared terms', text)
        self.assertIn('Rule: pre-specified', text)

    def test_md_table_escapes_pipes(self):
        self.assertIn('a/b', sm.md_table(['h'], [['a|b']]))


if __name__ == '__main__':
    unittest.main()
