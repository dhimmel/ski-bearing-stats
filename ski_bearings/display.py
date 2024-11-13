import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import htmltools
import IPython.display
import polars as pl
import reactable
from mizani.palettes import gradient_n_pal

from ski_bearings.analyze import (
    get_display_ski_area_filters,
    load_bearing_distribution_pl,
    load_ski_areas_pl,
)
from ski_bearings.utils import get_data_directory


def export_display_notebook() -> None:
    directory = get_data_directory()
    subprocess.check_call(
        args=[
            "jupyter",
            "nbconvert",
            "--execute",
            "--to=html",
            f"--output-dir={directory}",
            "--no-input",
            Path(__file__).parent.joinpath("display.ipynb"),
        ],
    )


def embed_reactable_html() -> None:
    IPython.display.display(IPython.display.HTML(html_style))
    reactable.embed_css()


def country_code_to_emoji(country_code: str) -> str:
    assert len(country_code) == 2
    return unicodedata.lookup(
        f"REGIONAL INDICATOR SYMBOL LETTER {country_code[0]}"
    ) + unicodedata.lookup(f"REGIONAL INDICATOR SYMBOL LETTER {country_code[1]}")


def get_ski_area_frontend_table() -> pl.DataFrame:
    """
    Dataframe of ski areas and their metrics designed for display.
    """
    cardinal_direction_props = (
        load_bearing_distribution_pl()
        .filter(pl.col("num_bins") == 4)
        .pivot(on="bin_label", index="ski_area_id", values="bin_proportion")
        .select(
            "ski_area_id",
            pl.all().exclude("ski_area_id").round(5).name.prefix("bin_proportion_"),
        )
    )
    return (
        load_ski_areas_pl()
        .filter(*get_display_ski_area_filters())
        .select(
            "ski_area_id",
            "ski_area_name",
            "status",
            pl.col("country_code")
            .map_elements(country_code_to_emoji, return_dtype=pl.String)
            .alias("country_emoji"),
            "country",
            "country_code",
            "region",
            "locality",
            "run_count_filtered",
            pl.col("combined_vertical").round(5),
            "bearing_mean",
            "bearing_alignment",
            "poleward_affinity",
            "eastward_affinity",
            pl.format("""<img src="ski_areas/{}.svg">""", "ski_area_id").alias("rose"),
        )
        .join(cardinal_direction_props, on="ski_area_id", how="left")
        .sort("ski_area_name")
    )


_country_filter = reactable.JS("""
function(rows, columnId, filterValue) {
    const filterValueLower = filterValue.toLowerCase();
    return rows.filter(function(row) {
        return (
            (row.values["country"] && row.values["country"].toLowerCase().includes(filterValueLower)) ||
            (row.values["country_code"] && row.values["country_code"].toLowerCase() === filterValueLower) ||
            (row.values["country_emoji"] && row.values["country_emoji"] === filterValue)
        );
    });
}
""")

_min_value_filter = reactable.JS("""
function(rows, columnId, filterValue) {
    return rows.filter(function(row) {
        return (
            row.values[columnId] >= filterValue
        );
    });
}
""")

_min_percent_filter = reactable.JS("""
function(rows, columnId, filterValue) {
    return rows.filter(function(row) {
        return (
            100 * row.values[columnId] >= filterValue
        );
    });
}
""")

_sequential_percent_palette = gradient_n_pal(["#ffffff", "#a100bf"])
_diverging_percent_palette = gradient_n_pal(["#e89200", "#ffffff", "#007dbf"])


def _percent_sequential_style(ci: reactable.CellInfo) -> dict[str, Any] | None:
    """Style cell background for columns whose values range from 0% to 100%."""
    if not isinstance(ci.value, int | float):
        return None
    color = _sequential_percent_palette(ci.value)
    return {"background": color}


def _percent_diverging_style(ci: reactable.CellInfo) -> dict[str, Any] | None:
    """Style cell background for columns whose values range from -100% to 100%."""
    if not isinstance(ci.value, int | float):
        return None
    color = _diverging_percent_palette((ci.value + 1) / 2)
    return {"background": color}


# derived from example https://machow.github.io/reactable-py/demos/twitter-followers.html
_percent_with_bar_cell = reactable.JS("""
function(cellInfo) {
    // Format as percentage
    const pct = (cellInfo.value * 100).toFixed(0) + "%"
    // Pad single-digit numbers
    let value = pct.padStart(4)
    // Render bar chart
    return `
    <div class="bar-cell">
        <span class="number">${value}</span>
        <div class="bar-chart" style="background-color: #e1e1e1">
        <div class="bar" style="width: ${pct}; background-color: #fc5185"></div>
        </div>
    </div>
    `
}
""")

_azimuth_cell = reactable.JS("""
function(cellInfo) {
    const azimuth = cellInfo.value; // Original azimuth for arrow rotation
    const displayedAzimuth = Math.round(azimuth); // Rounded azimuth for display only

    return `
    <div class="azimuth-arrow-cell" style="display: flex; align-items: center; justify-content: center; flex-direction: column;">
        <svg width="24" height="24" viewBox="0 0 24 24" style="transform: rotate(${azimuth}deg); margin-bottom: 4px;">
            <circle cx="12" cy="12" r="2" fill="black" />
            <line x1="12" y1="12" x2="12" y2="4" stroke="black" stroke-width="2" />
            <polygon points="8,6 12,1 16,6" fill="black" />
        </svg>
        <span style="font-size: 12px; color: #333;">${displayedAzimuth}°</span>
    </div>
    `;
}
""")


def get_ski_area_reactable() -> reactable.Reactable:
    data_pl = get_ski_area_frontend_table()

    def _ski_area_cell(ci: reactable.CellInfo) -> htmltools.Tag:
        ski_area_id = data_pl.item(row=ci.row_index, column="ski_area_id")
        url = htmltools.a(
            ci.value,
            href=f"https://openskimap.org/?obj={ski_area_id}",
            target="blank_",
        )
        return url

    def _country_cell(ci: reactable.CellInfo) -> str:
        country_emoji = data_pl.item(row=ci.row_index, column="country_emoji")
        return f"{country_emoji}<br>{ci.value}"

    bin_proportion_column_kwargs = {
        "format": reactable.ColFormat(percent=True, digits=0),
        "max_width": 45,
        "filterable": True,
        "filter_method": _min_percent_filter,
        "style": _percent_sequential_style,
    }

    return reactable.Reactable(
        data=data_pl,
        striped=True,
        searchable=False,
        highlight=True,
        full_width=True,
        columns=[
            reactable.Column(
                id="ski_area_id",
                show=False,
            ),
            reactable.Column(
                id="ski_area_name",
                name="Ski Area",
                cell=_ski_area_cell,
                min_width=250,
                filterable=True,
                sticky="left",  # makes entire group sticky
            ),
            reactable.Column(
                id="status",
                name="Status",
                show=False,
            ),
            reactable.Column(
                id="country",
                name="Country",
                cell=_country_cell,
                html=True,
                filter_method=_country_filter,
                filterable=True,
            ),
            reactable.Column(
                id="country_emoji",
                show=False,
            ),
            reactable.Column(
                id="country_code",
                show=False,
            ),
            reactable.Column(
                id="region",
                name="Region",
                filterable=True,
            ),
            reactable.Column(
                id="locality",
                name="Locality",
                filterable=True,
            ),
            reactable.Column(
                id="run_count_filtered",
                name="Runs",
                filterable=True,
                filter_method=_min_value_filter,
            ),
            reactable.Column(
                id="combined_vertical",
                name="Vertical",
                format=reactable.ColFormat(suffix="m", digits=0, separators=True),
                filterable=True,
                filter_method=_min_value_filter,
            ),
            reactable.Column(
                id="bearing_mean",
                name="Azimuth",
                # format=reactable.ColFormat(suffix="°", digits=0),
                cell=_azimuth_cell,
                html=True,
            ),
            reactable.Column(
                id="bearing_alignment",
                name="Alignment",
                # format=reactable.ColFormat(percent=True, digits=0),
                cell=_percent_with_bar_cell,
                html=True,
                filterable=True,
                filter_method=_min_percent_filter,
            ),
            reactable.Column(
                id="poleward_affinity",
                name="Poleward",
                format=reactable.ColFormat(percent=True, digits=0),
                filterable=True,
                filter_method=_min_percent_filter,
                style=_percent_diverging_style,
            ),
            reactable.Column(
                id="eastward_affinity",
                name="Eastward",
                format=reactable.ColFormat(percent=True, digits=0),
                filterable=True,
                filter_method=_min_percent_filter,
                style=_percent_diverging_style,
            ),
            reactable.Column(
                id="bin_proportion_N",
                name="N",
                **bin_proportion_column_kwargs,
            ),
            reactable.Column(
                id="bin_proportion_E",
                name="E",
                **bin_proportion_column_kwargs,
            ),
            reactable.Column(
                id="bin_proportion_S",
                name="S",
                **bin_proportion_column_kwargs,
            ),
            reactable.Column(
                id="bin_proportion_W",
                name="W",
                **bin_proportion_column_kwargs,
            ),
            reactable.Column(
                id="rose",
                name="Rose",
                html=True,
                # max_width=45,
            ),
        ],
        column_groups=[
            reactable.ColGroup(name="Ski Area", columns=["ski_area_hyper", "status"]),
            reactable.ColGroup(
                name="Location",
                columns=[
                    "country_emoji",
                    "country",
                    "region",
                    "locality",
                    "country_subdiv_code",
                ],
            ),
            reactable.ColGroup(
                name="Downhill Runs",
                columns=["run_count_filtered", "combined_vertical"],
            ),
            reactable.ColGroup(
                name="Mean Bearing",
                columns=[
                    "bearing_mean",
                    "bearing_alignment",
                    "poleward_affinity",
                    "eastward_affinity",
                ],
            ),
            reactable.ColGroup(
                name="Cardinal Directions",
                columns=[
                    "bin_proportion_N",
                    "bin_proportion_E",
                    "bin_proportion_S",
                    "bin_proportion_W",
                ],
            ),
        ],
    )


html_style = """
<style>
.number {
  font-family: "Fira Mono", Consolas, Monaco, monospace;
  font-size: 0.84375rem;
  white-space: pre;
}

.bar-chart {
  flex-grow: 1;
  margin-left: 0.375rem;
  height: 0.875rem;
}

.bar {
  height: 100%;
}
</style>
"""
