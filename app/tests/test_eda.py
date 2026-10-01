"""Regression checks for plotting helpers stored in the EDA notebook."""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class EdaNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "red_wine_quality_app"
            / "eda.ipynb"
        )
        if not notebook_path.is_file():
            raise unittest.SkipTest("EDA notebook is not included in this app copy")
        cls.notebook = json.loads(notebook_path.read_text())

    def plotting_function(self, name):
        namespace = {"np": np, "pd": pd, "plt": plt}
        source = next(
            "".join(cell["source"])
            for cell in self.notebook["cells"]
            if f"def {name}" in "".join(cell.get("source", []))
        )
        exec(source, namespace)
        return namespace[name]

    def tearDown(self):
        plt.close("all")

    def test_distribution_plot_accepts_one_row_of_subplots(self):
        with patch.object(plt, "show"):
            self.plotting_function("plotPerColumnDistribution")(
                pd.DataFrame({"quality": [3, 4, 5]}), 10, 5
            )
        self.assertEqual(len(plt.gcf().axes), 1)

    def test_correlation_plot_accepts_dataframe(self):
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 2, 3, 1]})
        frame.dataframeName = "sample.csv"
        with patch.object(plt, "show"):
            self.plotting_function("plotCorrelationMatrix")(frame, 4)
        self.assertGreater(len(plt.gcf().axes), 0)

    def test_scatter_plot_accepts_dataframe(self):
        frame = pd.DataFrame({"a": np.arange(20), "b": np.sin(np.arange(20))})
        with patch.object(plt, "show"):
            self.plotting_function("plotScatterMatrix")(frame, 4, 8)
        self.assertGreater(len(plt.gcf().axes), 0)
