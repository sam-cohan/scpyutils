"""
Plotting utilities.
"""


import datetime
import re
import sys
from typing import Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from formatutils import try_fmt_datetime, try_fmt_timedelta, try_fmt_num
from .statsutils import weighted_percentile

CM_COOLWARM = mpl.cm.coolwarm  # pylint: disable=no-member


def add_value_labels(
    ax: mpl.axes.Axes,
    spacing: int = 5,
    format_str: str = "{:,.2f}"
):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
        format_str (str): format to be used for the values.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = format_str.format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points",  # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
        # positive and negative values.


def plot_surface(
    df,
    title="",
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
):
    """Given a pandas DataFrame plot is a surface."""
    from mpl_toolkits.mplot3d import Axes3D

    x = list(range(len(df.T.index)))
    y = list(range(len(df.index)))
    X, Y = np.meshgrid(x, y)
    Z = df.values

    def get_lims(xx, min_margin=0.05, max_margin=0.05):
        min_, max_ = np.min(xx), np.max(xx)
        len_ = max_ - min_
        return min_ - len_ * min_margin, max_ + len_ * max_margin

    min_x, max_x = get_lims(x)
    min_y, max_y = get_lims(y)
    min_z, max_z = get_lims(Z)

    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    alpha=0.25, cmap=CM_COOLWARM)
    ax.contourf(X, Y, Z, zdir='x', offset=min_x, cmap=CM_COOLWARM)
    ax.contourf(X, Y, Z, zdir='y', offset=min_y, cmap=CM_COOLWARM)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_xlim((min_x, max_x))
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_ylim((min_y, max_y))
    ax.set_zlabel(zlabel, fontweight="bold")
    ax.set_zlim((min_z, max_z))
    ax.view_init(35, 35)
    ax.set_title(title, fontweight="bold")

    def shorten_if_date(x):
        x = str(x)
        match = re.match(r"(^\d\d\d\d-\d\d-\d\d).*", x)
        return match.groups()[0] if match else x

    x_skip = int(len(x) / 20) if len(x) > 30 else 1
    ax.xaxis.set_ticks(x[::x_skip])
    ax.xaxis.set_ticklabels([shorten_if_date(x) for x in df.T.index.get_level_values(-1)[::x_skip]],
                            rotation=90, alpha=0.8)
    y_skip = int(len(y) / 20) if len(y) > 30 else 1
    ax.yaxis.set_ticks(y[::y_skip])
    ax.yaxis.set_ticklabels([shorten_if_date(x) for x in df.index.get_level_values(-1)[::y_skip]],
                            rotation=90, alpha=0.8)

    return ax


def convert_ax_labels_from_secs_to_timedelta(ax, include_secs=False):
    plt.draw()  # make sure the tick labels are drawn otherwise it will be empty
    return ax.set_ticklabels(
        [
            "{}{}".format(
                try_fmt_timedelta(x.get_text(), full_precision=False),
                (" ({} s)".format(try_fmt_num(x.get_text())) if include_secs else ""))
            for x in ax.get_ticklabels()
        ],
        rotation=90)


def convert_ax_labels_from_secs_to_datetime(ax, include_secs=False):
    plt.draw()  # make sure the tick labels are drawn otherwise it will be empty
    return ax.set_ticklabels(
        [
            "{}{}".format(
                try_fmt_datetime(x.get_text()),
                (" ({} s)".format(try_fmt_num(x.get_text())) if include_secs else ""))
            for x in ax.get_ticklabels()
        ],
        rotation=90)


def set_label_formatter(ax, x_or_y, formatter):
    if not formatter:
        return
    if isinstance(formatter, str):
        if formatter == "timedelta":
            return lambda x, _: try_fmt_timedelta(x, full_precision=False)
        elif formatter == "datetime":
            return lambda x, _: try_fmt_datetime(x)
        elif formatter == "numeric":
            return lambda x, _: try_fmt_num(x)
        elif formatter == "percent":
            return lambda x, _: "{:,.2f}%".format(x * 100)
        elif formatter == "percent_score":
            return lambda x, _: "{:,.2f}%".format(x)
        elif formatter == "prob":
            return lambda x, _: "{:,.2f}".format(x)
        else:
            raise Exception("unknown formatter type {}".format(formatter))
    else:
        if not isinstance(formatter, mpl.ticker.FuncFormatter):
            formatter = mpl.ticker.FuncFormatter(formatter)
        axis = getattr(ax, "get_{}axis".format(x_or_y))()
        axis.set_major_formatter(formatter)
        axis.set_minor_formatter(formatter)


def annotate_plot(
    ax,
    txt,
    x_rel=0.4,
    y_rel=0.5,
    color="black",
    alpha=1.0,
    bbox_color="white",
    bbox_alpha=0.75,
    horizontalalignment="center",
    verticalalignment="center",
):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xstep = ((xmax - xmin) * 0.01)
    ystep = ((ymax - ymin) * 0.01)
    return ax.text(
        xmin + x_rel * 100 * xstep,
        ymin + y_rel * 100 * ystep,
        txt,
        color=color,
        alpha=alpha,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        bbox={"alpha": bbox_alpha, "color": bbox_color})


def _get_axes(*args, **kwargs):  # pylint: disable=unused-argument
    ax = kwargs.get("ax")
    if ax is None:
        fig = kwargs.get("fig")
        if fig is None:
            fig = plt.figure(figsize=kwargs.get("figsize", (12, 8)))
        ax = plt.subplot(111)
    title = kwargs.get("title")
    if title is not None:
        ax.set_title(
            title,
            color=kwargs.get("title_color", "black"),
            fontdict=dict(
                fontsize=kwargs.get("title_fontsize", 12)
            )
        )

    for x_or_y in ["x", "y"]:
        label = kwargs.get("{}label".format(x_or_y))
        if label is not None:
            getattr(ax, "set_{}label".format(x_or_y))(label)
        lim = kwargs.get("{}lim".format(x_or_y))
        if lim is not None:
            getattr(ax, "set_{}lim".format(x_or_y))(lim)
        formatter = kwargs.get("{}label_formatter".format(x_or_y),
                               kwargs.get("{}axis_type".format(x_or_y)))
        if formatter is not None:
            set_label_formatter(ax, x_or_y, formatter)

    ax2 = kwargs.get("ax2")
    y2label = kwargs.get("y2label")
    if y2label is not None:
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.set_ylabel(y2label)
        xlim = kwargs.get("xlim")
        if xlim is not None:
            ax2.set_xlim(xlim)
    y2lim = kwargs.get("y2lim")
    if y2lim:
        if ax2 is None:
            ax2 = ax.twinx()
        ax2.set_ylim(y2lim)
    if ax2 is not None:
        formatter = kwargs.get("y2label_formatter", kwargs.get("y2axis_type"))
        set_label_formatter(ax2, "y", formatter)
    return (ax, ax2)


def plot(xs, ys, *args, **kwargs):
    """Wrapper for matplotlib's plot and steps function.

    Arguments:
        xs: bottom axis values
        ys: right axis values
        ax (matplotlib.axes.Axes): matplotlib axis in case you want to plot
            on an existing axis (defaults to None and will create a new
            figure and axis).
        ax2 (matplotlib.axes.Axes): same as ax but for right axis.
    """
    # since pandas timestamps are not properly supported, turn them into datetime.datetime
    if pd.api.types.is_datetime64_any_dtype(xs) \
            or np.all([pd.isnull(x) or isinstance(x, pd.Timestamp) for x in xs]):
        xs = np.array([pd.to_datetime(x).to_pydatetime() for x in xs])
        if not kwargs.get("xlim"):
            kwargs["xlim"] = [np.min(xs), np.max(xs)]

    if isinstance(xs, pd.Series):
        xs = xs.values
    if isinstance(ys, pd.Series):
        ys = ys.values

    ax, ax2 = _get_axes(*args, **kwargs)

    # keep ordinal types as they are
    if np.all([pd.api.types.is_number(x) for x in xs]) \
            or np.all([isinstance(x, datetime.datetime) for x in xs]):
        point_labels = kwargs.get("pointlabels")
    else:  # treat everything else as categorical
        point_labels = xs
        xs = list(range(len(xs)))

    color = kwargs.get("color", "b")
    if kwargs.get("y2cum"):
        if not ax2:
            raise Exception(
                "WARINNG: no ax2 available to plot y2cum (make sure you provide ax2 or set y2label)")
        y2s = np.cumsum(ys)
        if kwargs.get("cum_normed", True):
            y2s /= np.sum(ys)
        ax2.step(
            xs, y2s,
            color=kwargs.get("y2color", "cyan"),
            alpha=kwargs.get("y2alpha", 0.5),
            marker=kwargs.get("y2marker", ""),
            where="post",
            label=kwargs.get("y2legendlabel", "cumulative_" +
                             kwargs.get("ylabel", ""))
        )

    plot_kwargs = {
        "alpha": kwargs.get("alpha", 0.8),
        "marker": kwargs.get("marker", "x"),
        "label": kwargs.get("legendlabel", ""),
    }
    plot_kwargs.update(
        {k: v for k, v in kwargs.items()
         if k in ["c", "cmap", "color",
                  "markeredgecolor", "markersize",
                  "linestyle"]})

    plot_type = kwargs.get("plot_type", "step")
    if plot_type == "step":
        plot_kwargs["where"] = kwargs.get("where", "post")
    elif plot_type == "bar":
        plot_kwargs = {
            k: v for k, v in plot_kwargs.items()
            if k not in {"cmap", "c", "marker"}}

    # since cmap is useful but only works with scatter, if you provide the cmap, then
    # we will force a scatter plot which has the cmap to overlay you plot
    if plot_type != "scatter" and "cmap" in plot_kwargs:
        ax.scatter(
            xs, ys,
            s=kwargs.get("markersize", 50),
            edgecolor=plot_kwargs.get("markeredgecolor", "none"),
            **{k: v for k, v in plot_kwargs.items()
               if k not in {"where", "color", "markeredgecolor", "markersize"}})
        plot_kwargs = {
            k: v for k, v in plot_kwargs.items()
            if k not in {"cmap", "c", "marker", "label"}}

    res = getattr(ax, plot_type)(xs, ys, **plot_kwargs)
    if plot_type == "bar" and point_labels is not None:
        for i, xl in enumerate(point_labels):
            rect = res[i]
            height = kwargs.get("pointlabel_height")
            if height is None:
                height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.,
                1.05 * height,
                "{} ({})".format(xl, try_fmt_num(rect.get_height())),
                ha='center', va='bottom', rotation='vertical')
            rect.set_color(color)

    plt.setp(ax.get_xticklabels(),
             rotation=kwargs.get("xlabel_rotation", 90),
             fontsize=kwargs.get("xlabel_fontsize", 10))
    plt.setp(ax.get_yticklabels(), color=color)
    ax.yaxis.label.set_color(color)
    if ax2:
        plt.setp(ax2.get_yticklabels(), color=color)
        ax2.yaxis.label.set_color(color)
    return res, (ax, ax2)


def plot_percentiles(ax, vals, weights=None, clip_min=None, clip_max=None, color=None, alpha=None):
    """Given a matplotlib axis and a series of values, calculate and plot
    vertical lines for some useful percentiles.

    10th and 90th percentiles with dots
    25th and 75th percentiles with dash-dots
    50th percentile with dashes

    Arguments:
        ax (matplotlib.axes.Axes): matplotlib axis
        vals: array like of values.
        weights: array-like of the same length as vals (defaults to None
            meaning equal weights)
        clip_min: numeric indicating if the plotted values were clipped from
            below.
        clip_max: numeric indicating if the plotted values were clipped from
            above.
    """
    percentiles = [10, 25, 50, 75, 90]
    color = "red" if color is None else color
    alpha = 0.6 if alpha is None else alpha
    pctl_vals = {
        pctl: pctl_val
        for pctl, pctl_val in zip(percentiles, weighted_percentile(vals, percentiles, weights))}

    def is_in_range(x):
        return (clip_min is None or x >= clip_min) and (clip_max is None or x <= clip_max)

    if is_in_range(pctl_vals[10]):
        ax.axvline(pctl_vals[10], linestyle=":", color=color, alpha=alpha)
    if is_in_range(pctl_vals[90]):
        ax.axvline(pctl_vals[90], linestyle=":", color=color, alpha=alpha)
    if is_in_range(pctl_vals[25]):
        ax.axvline(pctl_vals[25], linestyle="-.", color=color, alpha=alpha)
    if is_in_range(pctl_vals[75]):
        ax.axvline(pctl_vals[75], linestyle="-.", color=color, alpha=alpha)
    if is_in_range(pctl_vals[50]):
        ax.axvline(pctl_vals[50], linestyle="--", color=color, alpha=alpha)


PRESET_TABLE_BBOXES = {
    # (x_axis_left_border, y_axis_bottom_border, width, height)
    "center": (0.35, 0.2, 0.3, 0.79),
    "left": (0.1, 0.2, 0.3, 0.79),
    "right": (0.65, 0.2, 0.3, 0.79),
}


def plot_hist(vals, weights=None, **kwargs):
    """Wrap matplotlib's hist in a function that makes it easier to use.

    Arguments:
        vals: array-like of values to be used for histogram
        weights: optional array-like of weights corresponding to vals
        ax: matplotlib axis in case you want to plot on an existing axis
            (defaults to None and will create a new figure and axis)
        ax2: same as ax but for right axis
        ylabel: label for left y axis (defaults to "Frequency")
        cumulative: boolean indicating whether to plot the cumulative
            distribution on the y2 axis (default: False)
        show_stats_table: boolean indicating whether to display the stats
            summary table on the plot (default: True)
        percentiles: list of percentiles to calculate for the
            percentile_table (defaults to top and bottom 1st and 5th as well
            as all the deciles)
        table_bbox: tuple for table bbox (x_axis_left_border,
            y_axis_bottom_border, width, height). You can also
            provide a preset string value from "PRESET_TABLE_BBOXES"
            {'center', 'left', 'right'}
    """
    kwargs["ylabel"] = kwargs.get("ylabel", "Frequency")
    if kwargs.get("cumulative"):
        kwargs["y2label"] = kwargs.get("y2label", "Cumulative Frequency")
        kwargs["y2color"] = kwargs.get("color", "b")
    ax, ax2 = _get_axes(**kwargs)
    color = kwargs.get("color", "b")
    vals = np.array(vals)
    nulls = np.isnan(vals)
    num_nulls = np.sum(nulls)
    if num_nulls > 0:
        vals = vals[~nulls]
        print("WARNING: dropped %s NaN values... %s vals left to plot" % (
            num_nulls, len(vals)))
    if len(vals) == 0:
        print("WARNING: nothing to plot")
        return None, (ax, ax2)
    clip_min = kwargs.get("clip_min", -sys.maxsize)
    clip_max = kwargs.get("clip_max", sys.maxsize)
    alpha = kwargs.get("alpha", 0.5)

    plot_percentiles(
        ax,
        vals,
        weights,
        clip_min=clip_min,
        clip_max=clip_max,
        color=color,
        alpha=kwargs.get("percentiles_alpha", alpha)
    )
    clipped_vals = vals
    if (clip_min not in (None, -sys.maxsize)) or (clip_max not in (None, sys.maxsize)):
        clipped_vals = np.clip(vals, clip_min, clip_max)
    hist_args = dict(
        bins=kwargs.get("bins", 100),
        density=kwargs.get("normed", (weights is None)),
        label=kwargs.get("label", ""),
        color=color,
        alpha=alpha,
    )
    if weights is not None:
        weights = np.array(weights)
        if num_nulls > 0:
            weights = weights[~nulls]
        hist_args["weights"] = weights
    res = ax.hist(
        clipped_vals,
        **hist_args
    )
    if kwargs.get("cumulative", False):
        hist_args["cumulative"] = kwargs.get("y2cum", True)
        hist_args["density"] = kwargs.get("y2normed", True)
        hist_args["color"] = kwargs.get("y2color", "r")
        ax2.hist(
            clipped_vals,
            histtype="step",
            **hist_args
        )

    plt.setp(ax.get_xticklabels(which="both"),
             rotation=kwargs.get("xlabel_rotation", 90),
             fontsize=kwargs.get("xlabel_fontsize", 10))
    if kwargs.get("show_stats_table", True):
        table_bbox = kwargs.get("table_bbox", "center")
        if not isinstance(table_bbox, tuple):
            table_bbox = PRESET_TABLE_BBOXES[table_bbox]
        percentiles = kwargs.get(
            "percentiles", [1, 5] + list(np.linspace(10, 90, 9)) + [95, 99])
        count = len(vals)
        sample_rate = kwargs.get("sample_rate", 1.0)
        count_str = "~{:,.0f}".format(
            count / sample_rate) if sample_rate != 1.0 else "{:,.0f}".format(count)
        dist_summary = [
            ("count", count_str),
            ("mean", np.average(vals, weights=weights if weights is not None else None)),
            ("min", np.min(vals)),
        ] + [
            ("{:.0f}%".format(pctl), pctl_val)
            for pctl, pctl_val in zip(percentiles, weighted_percentile(vals, percentiles, weights))
        ] + [("max", np.max(vals))]
        xlabel_formatter = kwargs.get("xlabel_formatter", lambda x, p: str(x))
        tbl = ax.table(
            cellText=list(zip(*[[xlabel_formatter(v, k)
                                 if k != "count" else str(v)
                                 for k, v in dist_summary]])),
            rowLabels=[x[0] for x in dist_summary],
            colLabels=[kwargs.get(
                "table_title", kwargs.get("xlabel", "") + " Stats")],
            colWidths=[0.25],
            bbox=table_bbox,
        )
        tbl._cells[0, 0]._text.set_color(
            kwargs.get("table_title_color", kwargs.get("color", "black")))  # pylint: disable=protected-access
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(kwargs.get("table_fontsize", 12))

    return res, (ax, ax2)


def compare_hists(
    df: pd.DataFrame,
    filter_func: Callable,
    field: str,
    weight_fld: Optional[str] = None,
    label_for_filter: Optional[str] = None,
    label_for_complement: Optional[str] = None,
    **kwargs
):
    """Compare two histograms.

    Arguments:
        df (pd.DataFrame)L Dataframe of raw data.
        filter_func (Callable): a function that takes the df and for returns
            a series of booleans indicating whether the row at that index
            should be kept or not.
        field (str): name of the field used for plotting histogram.
        label_for_filter (str): label to be used in the legend for the series
            which passes the filter.
        label_for_complement (str): label to be used in the lenged for the
            series which does not pass filter.
    """
    kwargs = {
        k: w for k, w in kwargs.items()
        if k not in [
            "color", "label", "weights",
            "table_bbox", "table_title", "table_title_color",
        ]
    }
    if label_for_filter is None:
        import inspect
        label_for_filter = inspect.getsource(filter_func)
    if label_for_complement is None:
        label_for_complement = "NOT (%s)" % label_for_filter
    if "title" not in kwargs:
        kwargs["title"] = "{} distribution".format(field)
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Frequency"
    if "show_stats_table" not in kwargs:
        kwargs["show_stats_table"] = True

    filtered_df = df[filter_func(df)]
    _, (ax, ax2) = plot_hist(
        filtered_df[field],
        weights=filtered_df[weight_fld] if weight_fld is not None else None,
        label=label_for_filter,
        color="r",
        table_bbox=(0.2, 0.2, 0.3, 0.79),
        table_title=label_for_filter,
        table_title_color="r",
        **kwargs
    )
    complement_df = df[~filter_func(df)]
    plot_hist(
        complement_df[field],
        weights=complement_df[weight_fld] if weight_fld is not None else None,
        ax=ax,
        color="b",
        label=label_for_complement,
        table_bbox=(0.6, 0.2, 0.3, 0.79),
        table_title=label_for_complement,
        table_title_color="b",
        **kwargs
    )
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    leg.get_frame().set_alpha(0.5)
    return ax, ax2


def plot_marked_with_cum(
    df,
    y_fld,
    x_fld=None,
    sort_fld=None,
    ascending=False,
    marked_ids=None,
    **kwargs
):
    """Plot y_fld against x_fld for data which is sorted by sort_fld. second
    y axis will contain the normalized cumulative y_fld. Providing a list of
    ids as marked_ids will cause them to appear in red on the graph.

    This function is useful for visualizing if some variable tends to cluster
    around the sorted values of another variable (e.g. are most blocked
    source apps in the top spending apps?)

    Arguments:
        df (pd.DataFrame): DataFrame of the grouped data.
        y_fld (str): column name for y-axis data.
        x_fld (str): column name for x-axis data. If None, then it is assumed
            the x-axis will just be integer indexes starting at zero (defaults to None)
        sort_fld (str): columne name to sort by before plotting to see how
            the cumulative of the y-data changes as a function of changing
            sort_fld (defaults to None which means do not sort).
        ascending (bool): whether the sort should be ascending (default to
            False i.e. descending).
        marked_ids (set): set of ids which should be values from the df index
            and if a match will cause red dot to appear on the point on the graph.
        xlim (Tuple): x limits (default to None which means show full range).
        ylim (Tuple): y limits (defaults to None which means show full range).
        xlabel (str): label on x-axis (defaults to x_fld or 'index').
        ylabel (str): label on left y-axis (defaults to y_fld).
        y2label (str): label on right y-axis
        color (str): color of left y-axis line
        y2color (str): Color of right y-axis cumulative line
        title (str): String title of plot (default to '<y_fld> and its cumulative')
    """
    if not marked_ids:
        marked_ids = set()
    if sort_fld:
        df = df.sort_values(by=sort_fld, ascending=ascending)
    xs = range(len(df)) if not x_fld else df[x_fld]
    sorted_by_str = " Sorted By {} {}".format(
        sort_fld, "Ascending" if ascending else "Descending") if sort_fld else ""
    y2normed = kwargs.get("y2normed", True)
    _, (ax, ax2) = plot(
        xs,
        df[y_fld],
        xaxis_type=kwargs.get("xaxis_type"),
        xlim=kwargs.get("xlim", (0, np.max(xs) * 1.05)),
        ylim=kwargs.get("ylim", (0, df[y_fld].max() * 1.05)),
        y2lim=kwargs.get(
            "y2lim", (0, 1.1 if y2normed else (df[y_fld].sum() * 1.05))),
        title=kwargs.get("title", "{} and its cumulative".format(y_fld)),
        xlabel=kwargs.get("xlabel", (x_fld or "index") + sorted_by_str),
        ylabel=kwargs.get("ylabel", y_fld),
        color=kwargs.get("color", "blue"),
        y2label=kwargs.get("y2label", "cumulative {}".format(y_fld)),
        y2color=kwargs.get("y2color", "purple"),
        y2normed=y2normed,
        y2cum=True,
        alpha=0.2
    )

    if marked_ids:
        lines = ax.get_lines()
        line = lines[0]
        x_min, x_max = ax.xaxis.get_view_interval()
        y_min, y_max = ax.yaxis.get_view_interval()
        xs_ys = list(
            zip(*[(x, y)
                  for (idx, (x, y)) in enumerate(zip(line.get_xdata(), line.get_ydata()))
                  if (x_min <= x <= x_max
                      ) and (y_min <= y <= y_max
                             ) and df.iloc[[idx]].index.values[0] in marked_ids]))
        if xs_ys:
            xs, ys = xs_ys
            ax.scatter(xs, ys, color="r", marker="o", label=kwargs.get(
                "marked_label", "Marked"), alpha=0.6)

    leg = ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
    leg.get_frame().set_alpha(0.5)
    leg = ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    leg.get_frame().set_alpha(0.5)
    return ax, ax2
