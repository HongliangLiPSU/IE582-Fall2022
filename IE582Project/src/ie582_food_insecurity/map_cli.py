from __future__ import annotations

import argparse
from pathlib import Path

from .visualization import MapConfig, create_county_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create interactive county-level food insecurity index map."
    )
    parser.add_argument(
        "--index-file",
        type=Path,
        default=MapConfig.index_file,
        help="CSV with county FIPS and index values.",
    )
    parser.add_argument(
        "--fips-col",
        type=str,
        default=MapConfig.fips_col,
        help="FIPS column in index file.",
    )
    parser.add_argument(
        "--index-col",
        type=str,
        default=MapConfig.index_col,
        help="Index value column in index file.",
    )
    parser.add_argument(
        "--missing-strategy",
        type=str,
        choices=["keep", "state_median", "national_median", "zero"],
        default=MapConfig.missing_strategy,
        help="How to handle counties missing index values.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["geo", "mapbox"],
        default=MapConfig.engine,
        help="Rendering engine. Use 'geo' first for IDE compatibility.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=MapConfig.output_html,
        help="Output interactive HTML map path.",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=MapConfig.output_data,
        help="Output merged map data CSV path.",
    )
    parser.add_argument(
        "--output-missing",
        type=Path,
        default=MapConfig.output_missing,
        help="Output CSV with counties still missing after strategy.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=MapConfig.output_summary,
        help="Output map summary JSON path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=MapConfig.title,
        help="Map title.",
    )
    parser.add_argument(
        "--geojson-cache",
        type=Path,
        default=MapConfig.geojson_cache,
        help="Local cache path for county GeoJSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MapConfig(
        index_file=args.index_file,
        fips_col=args.fips_col,
        index_col=args.index_col,
        missing_strategy=args.missing_strategy,
        engine=args.engine,
        geojson_cache=args.geojson_cache,
        output_html=args.output_html,
        output_data=args.output_data,
        output_missing=args.output_missing,
        output_summary=args.output_summary,
        title=args.title,
    )
    summary = create_county_map(config)
    print("Map generation complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
