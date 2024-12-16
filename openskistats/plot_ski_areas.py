import plotnine as pn
import polars as pl
from mizani.formatters import percent_format

from openskistats.analyze import load_ski_areas_pl
from openskistats.utils import gini_coefficient


def get_ski_area_metric_percentiles(
    ski_area_filters: list[pl.Expr] | None = None,
) -> pl.DataFrame:
    metrics = [
        "lift_count",
        "run_count",
        "coordinate_count",
        "segment_count",
        "combined_vertical",
        "combined_distance",
        "vertical_drop",
    ]
    return (
        load_ski_areas_pl(ski_area_filters)
        .unpivot(on=metrics, index=["ski_area_id", "ski_area_name"])
        .sort("variable", "value", "ski_area_id")
        .with_columns(
            pl.col("value").rank(method="ordinal").over("variable").alias("value_rank"),
            pl.col("value").cum_sum().over("variable").alias("value_cumsum"),
        )
        .with_columns(
            pl.col("value_cumsum")
            .truediv(pl.sum("value").over("variable"))
            .alias("value_cdf"),
            pl.col("value_rank")
            .truediv(pl.count("value").over("variable"))
            .alias("value_rank_pctl"),
        )
    )


def plot_ski_area_metric_percentiles(
    ski_area_filters: list[pl.Expr] | None = None,
) -> tuple[pn.ggplot, pn.ggplot]:
    cdf_metrics = get_ski_area_metric_percentiles(ski_area_filters)
    gini_df = (
        cdf_metrics.group_by("variable", maintain_order=True)
        .agg(
            gini=pl.col("value").map_batches(gini_coefficient, returns_scalar=True),
        )
        .with_columns(
            variable_label=pl.col("variable").str.replace("_", " ").str.to_titlecase()
        )
        .sort("gini")
    )
    metrics_enum = pl.Enum(gini_df["variable"])
    gini_df = gini_df.with_columns(variable=gini_df["variable"].cast(metrics_enum))
    lorenz_curves = (
        pn.ggplot(
            data=cdf_metrics.with_columns(
                variable=pl.col("variable").cast(metrics_enum)
            ),
            mapping=pn.aes(
                x="value_rank_pctl",
                y="value_cdf",
                color="variable",
            ),
        )
        + pn.geom_abline(intercept=0, slope=1, linetype="dashed")
        + pn.scale_x_continuous(
            name="Percentile",
            labels=percent_format(),
            expand=(0.01, 0.01),
        )
        + pn.scale_y_continuous(
            name="Cumulative Share",
            labels=percent_format(),
            expand=(0.01, 0.01),
        )
        + pn.scale_color_discrete()
        + pn.geom_path(show_legend=False)
        + pn.coord_equal()
        + pn.theme_bw()
        + pn.theme(figure_size=(4.2, 4))
    )
    gini_bars = (
        pn.ggplot(
            data=gini_df,
            mapping=pn.aes(
                x="variable", y="gini", fill="variable", label="variable_label"
            ),
        )
        + pn.geom_col(show_legend=False)
        + pn.scale_y_continuous(
            name="Gini Coefficient",
            labels=percent_format(),
            expand=(0, 0),
            limits=(0, 1),
        )
        + pn.geom_text(y=0.05, ha="left")
        + pn.scale_x_discrete(
            name="", limits=list(reversed(metrics_enum.categories)), labels=None
        )
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(figure_size=(3, 4))
    )
    return lorenz_curves, gini_bars
