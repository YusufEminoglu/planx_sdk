# -*- coding: utf-8 -*-
"""Pytest wrapper for the ported QGIS plugin engine tests."""

import os
import runpy


def test_engine_correctness():
    """Runs the engine unit checks inside the main process using runpy to capture coverage."""
    script_path = os.path.join(os.path.dirname(__file__), "run_engine_tests.py")

    try:
        runpy.run_path(script_path, run_name="__main__")
    except SystemExit as e:
        assert e.code == 0, f"Engine tests failed with exit code {e.code}"
