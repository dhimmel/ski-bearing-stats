import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import htmltools
import IPython.display
import polars as pl
import reactable
from mizani.palettes import gradient_n_pal

from openskistats.analyze import (
    get_display_ski_area_filters,
    load_bearing_distribution_pl,
    load_ski_areas_pl,
)
from openskistats.models import SkiAreaModel
from openskistats.plot import NARROW_SPACE
from openskistats.utils import get_data_directory


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
    for html_ in html_style, html_script:
        IPython.display.display(IPython.display.HTML(html_))
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
        .join(cardinal_direction_props, on="ski_area_id", how="left")
        # inconveniently, order of selection here defines reactable column order
        # https://github.com/glin/reactable/issues/172
        .select(
            "ski_area_id",
            "ski_area_name",
            "osm_status",
            pl.col("country_code")
            .map_elements(country_code_to_emoji, return_dtype=pl.String)
            .alias("country_emoji"),
            "country",
            "country_code",
            "region",
            "locality",
            "latitude",
            "run_count",
            "lift_count",
            "combined_vertical",
            "min_elevation",
            "max_elevation",
            "vertical_drop",
            "bearing_mean",
            "bearing_alignment",
            "poleward_affinity",
            "eastward_affinity",
            pl.selectors.starts_with("bin_proportion_"),
            pl.format("""<img src="ski_areas/{}.svg">""", "ski_area_id").alias("rose"),
        )
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


def _latitude_cell(ci: reactable.CellInfo) -> htmltools.Tag:
    latitude = ci.value
    background_color = _latitude_palette(abs(latitude) / 90)
    hemisphere_symbol = "ℕ" if latitude > 0 else "𝕊"

    return htmltools.span(
        [
            htmltools.span(
                hemisphere_symbol,
                style=htmltools.css(
                    font_size="1em",  # Font size for the hemisphere symbol
                    line_height="1em",  # Adjust line height for better spacing
                ),
            ),
            htmltools.span(
                f"{latitude:.1f}°",
                style=htmltools.css(
                    font_size="0.7em",  # Smaller font for latitude
                    line_height="1em",  # Adjust line height for proper spacing
                    margin_top="0.2em",  # Adds vertical distance between lines
                ),
            ),
        ],
        class_="badge",
        style=htmltools.css(
            background_color=background_color,
            color="white",
            width="3em",  # Set equal width and height for a circle
            height="3em",
            border_radius="50%",  # Maximum border radius for a circle
            display="inline-flex",  # Use flexbox for layout
            flex_direction="column",  # Stack items vertically
            align_items="center",  # Center items horizontally
            justify_content="center",  # Center items vertically
            text_align="center",  # Center text within each line
        ),
    )


_latitude_filter = reactable.JS(r"""
function(rows, columnId, filterValue) {
    return rows.filter(function(row) {
        const latitude = row.values["latitude"];
        const hemisphere = latitude > 0 ? "north" : "south";

        // Check if filterValue is entirely alphabetic
        const isAlphabetic = /^[a-zA-Z]+$/.test(filterValue);

        // Use matchesNumericFilter for numeric filter values
        if (!isAlphabetic) {
            return matchesNumericFilter(latitude, filterValue);
        }

        // Handle string filter values for "north" or "south"
        if (typeof filterValue === "string") {
            return hemisphere.includes(filterValue.toLowerCase());
        }

        // Default: include all rows if filterValue is invalid
        return true;
    });
}
""")

_numeric_filter = reactable.JS("""
function(rows, columnId, filterValue) {
    return rows.filter(row => matchesNumericFilter(row.values[columnId], filterValue));
}
""")

_percent_filter = reactable.JS("""
function(rows, columnId, filterValue) {
    return rows.filter(row => matchesNumericFilter(row.values[columnId] * 100, filterValue));
}
""")

_sequential_percent_palette = gradient_n_pal(["#ffffff", "#a100bf"])
_diverging_percent_palette = gradient_n_pal(["#e89200", "#ffffff", "#007dbf"])
_latitude_palette = gradient_n_pal(["#ffffff", "#000000"])


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

columns_descriptions = {
    # pre-populate descriptions from SkiAreaModel
    **{name: field.description for name, field in SkiAreaModel.model_fields.items()},
    # add custom descriptions including overwriting SkiAreaModel descriptions
    "latitude": "Hemisphere where the ski area is located, either ℕ for north or 𝕊 for south, "
    "as well as the latitude (φ) in decimal degrees.",
    "bin_proportion_N": "Proportion of vertical-weighted run segments oriented with a northern cardinal direction.",
    "bin_proportion_E": "Proportion of vertical-weighted run segments oriented with an eastern cardinal direction.",
    "bin_proportion_S": "Proportion of vertical-weighted run segments oriented with a southern cardinal direction.",
    "bin_proportion_W": "Proportion of vertical-weighted run segments oriented with a western cardinal direction.",
}


def _format_header(ci: reactable.HeaderCellInfo) -> htmltools.Tag | str:
    """
    Format header cell with tooltip provided by an <abbr> tag.
    FIXME: The title attribute is inaccessible to most keyboard, mobile, and screen reader users,
    so creating tooltips like this is generally discouraged.

    References for tooltip headers:
    https://machow.github.io/reactable-py/get-started/format-header-footer.html#headers
    https://glin.github.io/reactable/articles/cookbook/cookbook.html#tooltips
    https://github.com/glin/reactable/issues/220
    """
    column_id = ci.name
    column_name = ci.value
    if description := columns_descriptions.get(column_id):
        return htmltools.tags.abbr(column_name, title=description)
    return column_name


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
        "filter_method": _percent_filter,
        "style": _percent_sequential_style,
    }
    meters_column_kwargs = {
        "format": reactable.ColFormat(
            suffix=f"{NARROW_SPACE}m",
            digits=0,
            separators=True,
        ),
        "filterable": True,
        "filter_method": _numeric_filter,
    }
    return reactable.Reactable(
        data=data_pl,
        striped=True,
        searchable=False,
        highlight=True,
        full_width=True,
        default_col_def=reactable.Column(
            header=_format_header,
        ),
        show_page_size_options=True,
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
                id="osm_status",
                name="Status",
                show=False,
            ),
            reactable.Column(
                id="latitude",
                name=f"ℍ{NARROW_SPACE}φ",
                cell=_latitude_cell,
                filterable=True,
                filter_method=_latitude_filter,
                max_width=60,
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
                id="run_count",
                name="Runs",
                filterable=True,
                filter_method=_numeric_filter,
            ),
            reactable.Column(
                id="lift_count",
                name="Lifts",
                filterable=True,
                filter_method=_numeric_filter,
            ),
            reactable.Column(
                id="combined_vertical",
                name="Vertical",
                **meters_column_kwargs,
            ),
            reactable.Column(
                id="vertical_drop",
                name="Drop",
                **meters_column_kwargs,
            ),
            reactable.Column(
                id="min_elevation",
                name="Base Elev",
                **meters_column_kwargs,
            ),
            reactable.Column(
                id="max_elevation",
                name="Peak Elev",
                **meters_column_kwargs,
            ),
            reactable.Column(
                id="bearing_mean",
                name="Azimuth",
                # format=reactable.ColFormat(suffix="°", digits=0),
                cell=_azimuth_cell,
                html=True,
                filterable=True,
                filter_method=_numeric_filter,
            ),
            reactable.Column(
                id="bearing_alignment",
                name="Alignment",
                # format=reactable.ColFormat(percent=True, digits=0),
                cell=_percent_with_bar_cell,
                html=True,
                filterable=True,
                filter_method=_percent_filter,
            ),
            reactable.Column(
                id="poleward_affinity",
                name="Poleward",
                format=reactable.ColFormat(percent=True, digits=0),
                filterable=True,
                filter_method=_percent_filter,
                style=_percent_diverging_style,
            ),
            reactable.Column(
                id="eastward_affinity",
                name="Eastward",
                format=reactable.ColFormat(percent=True, digits=0),
                filterable=True,
                filter_method=_percent_filter,
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
                sortable=False,
                # max_width=45,
            ),
        ],
        column_groups=[
            reactable.ColGroup(
                name="Ski Area", columns=["ski_area_hyper", "osm_status"]
            ),
            reactable.ColGroup(
                name="Location",
                columns=[
                    "latitude",
                    "country_emoji",
                    "country",
                    "region",
                    "locality",
                    "country_subdiv_code",
                ],
            ),
            reactable.ColGroup(
                name="Downhill Runs",
                columns=[
                    "run_count",
                    "lift_count",
                    "combined_vertical",
                    "min_elevation",
                    "max_elevation",
                    "vertical_drop",
                ],
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


html_script = r"""
<script>
function matchesNumericFilter(value, filterValue) {
    filterValue = filterValue.trim();
    let match;

    // Handle "-" and "-0" as shorthand for <= 0
    if (filterValue === "-" || filterValue === "-0") {
        return value <= 0;
    }

    // Handle basic numbers (>= for positive, <= for negative)
    if ((match = filterValue.match(/^(-?\d+)$/))) {
        const threshold = parseFloat(match[1]);
        return threshold >= 0 ? value >= threshold : value <= threshold;
    }

    // Handle explicit ranges with brackets (inclusive) or parentheses (exclusive)
    if ((match = filterValue.match(/^([\[\(])\s*(-?\d+)?\s*,\s*(-?\d+)?\s*([\]\)])$/))) {
        const lower = match[2] !== undefined ? parseFloat(match[2]) : -Infinity;
        const upper = match[3] !== undefined ? parseFloat(match[3]) : Infinity;
        const lowerInclusive = match[1] === '['; // True if inclusive lower
        const upperInclusive = match[4] === ']'; // True if inclusive upper

        return (
            (lowerInclusive ? value >= lower : value > lower) &&
            (upperInclusive ? value <= upper : value < upper)
        );
    }

    // Default: invalid filterValue returns true
    return true;
}
</script>
"""

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
