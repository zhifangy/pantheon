#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Misc functions."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Union
from pathlib import Path
from plotly.graph_objs._figure import Figure
import nilearn.plotting as nlp

PathLike = Union[Path, str]
PlotlyFigure = Figure
StatMapView = nlp.html_stat_map.StatMapView
SurfaceView = nlp.html_surface.SurfaceView
