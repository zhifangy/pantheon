#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plotting function for plotly."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Optional, Union

from ..utils.typing import PlotlyFigure


def apply_style(
    fig: PlotlyFigure,
    height: Optional[Union[int, float]] = None,
    width: Optional[Union[int, float]] = None,
    base_font_size: int = 16,
    font_family: str = "Arial",
    font_color: str = "black",
    showline: bool = True,
    showgrid: bool = True,
    showtitle_x: bool = True,
    showtitle_y: bool = True,
    showlegend: bool = True,
    showlegend_title: bool = False,
    template: str = "plotly_white",
) -> PlotlyFigure:
    """Applys custom plotly style to figure.

    Args:
        fig: A Ploty Figure.
        height: Figure height. If it's None, use a aspect ratio 1.618.
        width: Figure width. If it's None, use a aspect ratio 1.618.
        base_font_size: Base font size.
        font_family: Font typeface.
        font_color: Font color.
        showline: Show axis line. Default: True.
        showgrid: Show grid in figure. Default: True.
        showtitle_x: Show x axis title. Default: True.
        showtitle_y: Show y axis title. Default: True.
        showlegend: Show figure legend. Default: True.
        showlegend_title: Show legned title. Default: False.
        template: Plotly default figure style template.

    Returns:
        A plotly Figure object.
    """

    # use plotly theme
    if template:
        fig.update_layout(template=template)
    # figure size
    if height or width:
        # use aspect ratio 1.618 if only one dimension is set
        if height is None:
            height = width / 1.618
        if width is None:
            width = height * 1.618
        fig.update_layout(autosize=False, height=height, width=width)
    # font color
    fig.update_layout(title_font_color=font_color, legend_font_color=font_color)
    fig.update_xaxes(dict(title_font_color=font_color, tickfont_color=font_color))
    fig.update_yaxes(dict(title_font_color=font_color, tickfont_color=font_color))
    # font size
    if base_font_size:
        fig.update_layout(font_size=base_font_size)
        # axes
        fig.update_xaxes(
            dict(
                title_font=dict(size=int(base_font_size * 1.25)),
                tickfont=dict(size=base_font_size),
            )
        )
        fig.update_yaxes(
            dict(
                title_font=dict(size=int(base_font_size * 1.25)),
                tickfont=dict(size=base_font_size),
            )
        )
        # title
        fig.update_layout(title=dict(font=dict(size=int(base_font_size * 1.5))))
        # legend
        fig.update_layout(legend=dict(font=dict(size=base_font_size)))
        # annotation
        fig.update_annotations(font=dict(size=base_font_size))
    # font family
    if font_family:
        fig.update_layout(font_family=font_family)
    # show axis line
    fig.update_xaxes(showline=showline, linewidth=2, linecolor="black", ticks="outside")
    fig.update_yaxes(showline=showline, linewidth=2, linecolor="black", ticks="outside")
    # show background grid
    fig.update_xaxes(showgrid=showgrid)
    fig.update_yaxes(showgrid=showgrid)
    # hide x, y axis title
    if not showtitle_x:
        fig.update_xaxes(title_text="")
    if not showtitle_y:
        fig.update_yaxes(title_text="")
    # show figure legend
    fig.update_layout(showlegend=showlegend)
    # hide figure legend title
    if not showlegend_title:
        fig.update_layout(legend_title_text="")

    return fig
