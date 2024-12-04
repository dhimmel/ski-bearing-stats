import polars as pl

from openskistats.analyze import load_runs_pl


def get_bearing_by_latitude_bin_metrics() -> pl.DataFrame:
    """
    Metrics for (latitude_bin, bearing_bin) pairs.
    """
    bin_pattern = r"\[(.+), (.+)\)"
    return (
        load_runs_pl()
        .filter(pl.col("run_uses").list.contains("downhill"))
        .explode("run_coordinates_clean")
        .unnest("run_coordinates_clean")
        .filter(pl.col("segment_hash").is_not_null())
        .select(
            "run_id",
            "segment_hash",
            # pl.col("latitude").round(0).cast(pl.Int32).alias("latitude"),
            pl.col("latitude")
            .cut(breaks=range(-90, 91, 2), left_closed=True)
            .alias("latitude_bin"),
            pl.col("bearing")
            .cut(breaks=range(0, 361, 2), left_closed=True)
            .alias("bearing_bin"),
            # pl.col("bearing").round(0).cast(pl.Int32).alias("bearing"),
            "distance_vertical_drop",
        )
        .group_by("latitude_bin", "bearing_bin")
        .agg(
            pl.count("segment_hash").alias("segment_count"),
            pl.col("distance_vertical_drop").sum().alias("combined_vertical").round(5),
        )
        .filter(pl.col("combined_vertical") > 0)
        .with_columns(
            latitude_bin_lower=pl.col("latitude_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=1)
            .cast(pl.Int32),
            latitude_bin_upper=pl.col("latitude_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=2)
            .cast(pl.Int32),
            bearing_bin_lower=pl.col("bearing_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=1)
            .cast(pl.Int32),
            bearing_bin_upper=pl.col("bearing_bin")
            .cast(pl.String)
            .str.extract(pattern=bin_pattern, group_index=2)
            .cast(pl.Int32),
        )
        .with_columns(
            latitude_bin_center=pl.mean_horizontal(
                "latitude_bin_lower", "latitude_bin_upper"
            ),
            bearing_bin_center=pl.mean_horizontal(
                "bearing_bin_lower", "bearing_bin_upper"
            ),
        )
        .with_columns(bearing_bin_center_radians=pl.col("bearing_bin_center").radians())
        .sort("latitude_bin", "bearing_bin")
        .collect()
    )
