"""Checks for the model interface and the Streamlit app flow."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from red_wine_quality_app.model import FEATURE_NAMES, load_models, predict_wine


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = load_models()
        cls.model = cls.models["CatBoost"]

    def test_training_schema_and_evaluation(self):
        self.assertEqual(set(self.models), {"CatBoost", "XGBoost", "LightGBM", "Random Forest"})
        self.assertEqual(self.model.feature_names, FEATURE_NAMES)
        self.assertEqual(len(self.model.features), 11)
        self.assertGreater(self.model.train_size, self.model.test_size)
        self.assertAlmostEqual(
            self.model.metrics.error_rate, 1 - self.model.metrics.accuracy
        )
        self.assertEqual(self.model.metrics.quality_scores, (3, 4, 5, 6, 7, 8))
        self.assertEqual(len(self.model.metrics.confusion_matrix), 6)
        self.assertEqual(
            sum(sum(row) for row in self.model.metrics.confusion_matrix),
            self.model.test_size,
        )
        self.assertEqual(sum(self.model.metrics.actual_counts), self.model.test_size)
        self.assertEqual(sum(self.model.metrics.predicted_counts), self.model.test_size)
        self.assertGreaterEqual(self.model.metrics.within_one_accuracy, self.model.metrics.accuracy)
        for feature in self.model.features:
            self.assertLessEqual(feature.minimum, feature.default)
            self.assertLessEqual(feature.default, feature.maximum)
        for model in self.models.values():
            self.assertEqual(model.feature_names, FEATURE_NAMES)
            self.assertEqual(sum(model.metrics.actual_counts), model.test_size)
            self.assertEqual(sum(model.metrics.predicted_counts), model.test_size)

    def test_prediction_accepts_mapping_in_any_order(self):
        sample = {feature.name: feature.default for feature in self.model.features}
        for model in self.models.values():
            result = predict_wine(model, sample)
            self.assertEqual(result, predict_wine(model, dict(reversed(list(sample.items())))))
            self.assertIn(result.score, model.metrics.quality_scores)
            self.assertEqual(result.error_rate, model.metrics.error_rate)
            self.assertEqual(result.mean_absolute_error, model.metrics.mean_absolute_error)

    def test_prediction_rejects_incomplete_or_invalid_input(self):
        sample = {feature.name: feature.default for feature in self.model.features}
        with self.assertRaisesRegex(ValueError, "Expected exactly"):
            predict_wine(self.model, {"alcohol": 10.2})
        sample["alcohol"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_wine(self.model, sample)


class StreamlitAppTests(unittest.TestCase):
    def test_tabs_sliders_and_predict_button(self):
        entry_point = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(entry_point, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Calculations", "Visualizations", "Predictions"],
        )
        self.assertEqual(len(app.slider), 11)
        self.assertEqual(app.selectbox[0].options, ["CatBoost", "XGBoost", "LightGBM", "Random Forest"])
        self.assertEqual(len(app.get("vega_lite_chart")), 2)
        self.assertFalse(app.success)

        self.assertEqual(app.button[0].label, "Predict quality")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("Predicted quality score:", app.success[0].value)
        first_estimate = next(metric.value for metric in app.metric if metric.label == "Estimated quality")
        next(slider for slider in app.slider if slider.label == "Alcohol").set_value(14.9).run()
        app.button[0].click().run()
        second_estimate = next(metric.value for metric in app.metric if metric.label == "Estimated quality")
        self.assertNotEqual(first_estimate, second_estimate)
        metric_labels = [metric.label for metric in app.metric]
        self.assertIn("Exact-score accuracy", metric_labels)
        self.assertIn("Mean absolute error", metric_labels)
        self.assertIn("Held out mean absolute error", metric_labels)
        self.assertIn("Held out exact-score error", metric_labels)
        app.selectbox[0].select("XGBoost").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.success)
        app.button[0].click().run()
        self.assertIn("Predicted quality score:", app.success[0].value)


if __name__ == "__main__":
    unittest.main()
