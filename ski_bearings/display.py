import subprocess
import unicodedata
from pathlib import Path

import IPython.display
import polars as pl
import reactable

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
            pl.format(
                "<a href='https://openskimap.org/?obj={}' target='_blank'>{}</a>",
                "ski_area_id",
                "ski_area_name",
            ).alias("ski_area_hyper"),
            "status",
            pl.col("country_code")
            .map_elements(country_code_to_emoji, return_dtype=pl.String)
            .alias("country_emoji"),
            "country",
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


# derived from example https://machow.github.io/reactable-py/demos/twitter-followers.html
_js_percent = reactable.JS("""
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

_js_azimuth_arrow = reactable.JS("""
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
    return reactable.Reactable(
        data=get_ski_area_frontend_table().drop("ski_area_id", "ski_area_name"),
        striped=True,
        searchable=True,
        highlight=True,
        full_width=True,
        columns=[
            reactable.Column(
                id="ski_area_hyper",
                name="Name",
                html=True,
                min_width=250,
                searchable=True,
                sticky="left",  # makes entire group sticky
            ),
            reactable.Column(
                id="status",
                name="Status",
                searchable=True,
                show=False,
            ),
            reactable.Column(
                id="country_emoji",
                name="Country",
                searchable=True,
            ),
            reactable.Column(
                id="country",
                name="Country",
                searchable=True,
                show=False,
            ),
            reactable.Column(
                id="region",
                name="Region",
                searchable=True,
            ),
            reactable.Column(
                id="locality",
                name="Locality",
                searchable=True,
            ),
            reactable.Column(
                id="run_count_filtered",
                name="Runs",
                searchable=False,
            ),
            reactable.Column(
                id="combined_vertical",
                name="Vertical",
                format=reactable.ColFormat(suffix="m", digits=0, separators=True),
                searchable=False,
            ),
            reactable.Column(
                id="bearing_mean",
                name="Azimuth",
                # format=reactable.ColFormat(suffix="°", digits=0),
                searchable=False,
                cell=_js_azimuth_arrow,
                html=True,
            ),
            reactable.Column(
                id="bearing_alignment",
                name="Alignment",
                # format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
                cell=_js_percent,
                html=True,
            ),
            reactable.Column(
                id="poleward_affinity",
                name="Poleward",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
            ),
            reactable.Column(
                id="eastward_affinity",
                name="Eastward",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
            ),
            reactable.Column(
                id="bin_proportion_N",
                name="N",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
                max_width=45,
            ),
            reactable.Column(
                id="bin_proportion_E",
                name="E",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
                max_width=45,
            ),
            reactable.Column(
                id="bin_proportion_S",
                name="S",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
                max_width=45,
            ),
            reactable.Column(
                id="bin_proportion_W",
                name="W",
                format=reactable.ColFormat(percent=True, digits=0),
                searchable=False,
                max_width=45,
            ),
            reactable.Column(
                id="rose",
                name="Rose",
                searchable=False,
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
