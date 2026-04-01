from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Tuple
from urllib.request import urlopen

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover - runtime env specific
    raise SystemExit(
        "Failed to import plotly. Install dependencies with:\n"
        "  pip install -r requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MissingStrategy = Literal["keep", "state_median", "national_median", "zero"]
MapEngine = Literal["geo", "mapbox"]


@dataclass
class MapConfig:
    index_file: Path = PROJECT_ROOT / "outputs" / "nationwide_food_insecurity_index.csv"
    fips_col: str = "fips"
    index_col: str = "food_insecurity_index"
    missing_strategy: MissingStrategy = "state_median"
    engine: MapEngine = "geo"
    geojson_url: str = (
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    )
    geojson_cache: Path = (
        PROJECT_ROOT / "data" / "raw" / "geojson" / "us_counties_fips.geojson"
    )
    output_html: Path = PROJECT_ROOT / "outputs" / "maps" / "county_index_map.html"
    output_data: Path = PROJECT_ROOT / "outputs" / "maps" / "county_index_map_data.csv"
    output_missing: Path = PROJECT_ROOT / "outputs" / "maps" / "missing_counties.csv"
    output_summary: Path = PROJECT_ROOT / "outputs" / "maps" / "map_summary.json"
    title: str = "U.S. County-Level Food Insecurity Index"
    subtitle: str = (
        "Data-driven county-level risk index (0–100 scale; higher indicates higher risk)"
    )


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return out


def _load_geojson(cache_path: Path, url: str) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    with urlopen(url) as response:
        geojson = json.load(response)
    cache_path.write_text(json.dumps(geojson), encoding="utf-8")
    return geojson


def _extract_county_metadata(geojson: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        fips = str(feature.get("id", "")).zfill(5)
        rows.append(
            {
                "FIPS": fips,
                "county_name": props.get("NAME", ""),
                "county_lsad": props.get("LSAD", ""),
                "state_fips": props.get("STATE", ""),
                "county_fips": props.get("COUNTY", ""),
            }
        )
    return pd.DataFrame(rows).drop_duplicates("FIPS")


def _prepare_index_data(path: Path, fips_col: str, index_col: str) -> Tuple[pd.DataFrame, int]:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")

    df = pd.read_csv(path)
    df = _clean_columns(df)

    if fips_col not in df.columns:
        raise ValueError(f"FIPS column '{fips_col}' not found in {path}. Columns: {list(df.columns)}")
    if index_col not in df.columns:
        raise ValueError(
            f"Index column '{index_col}' not found in {path}. Columns: {list(df.columns)}"
        )

    index_df = pd.DataFrame(
        {
            "FIPS": df[fips_col].astype(str).str.strip().str.zfill(5),
            "index_original": pd.to_numeric(df[index_col], errors="coerce"),
        }
    )

    dup_count = int(index_df.duplicated("FIPS").sum())
    index_df = (
        index_df.groupby("FIPS", as_index=False)["index_original"]
        .mean()
        .sort_values("FIPS")
        .reset_index(drop=True)
    )
    return index_df, dup_count


def _apply_missing_strategy(df: pd.DataFrame, strategy: MissingStrategy) -> pd.DataFrame:
    out = df.copy()
    out["missing_original"] = out["index_original"].isna()

    if strategy == "keep":
        out["index_for_map"] = out["index_original"]
    elif strategy == "national_median":
        national = float(out["index_original"].median(skipna=True))
        out["index_for_map"] = out["index_original"].fillna(national)
    elif strategy == "state_median":
        state_med = out.groupby("state_fips")["index_original"].transform("median")
        national = float(out["index_original"].median(skipna=True))
        out["index_for_map"] = out["index_original"].fillna(state_med).fillna(national)
    elif strategy == "zero":
        out["index_for_map"] = out["index_original"].fillna(0.0)
    else:
        raise ValueError(f"Unsupported missing strategy: {strategy}")

    out["imputed"] = out["missing_original"] & out["index_for_map"].notna()
    out["missing_after_strategy"] = out["index_for_map"].isna()
    return out


def _build_hover_text(df: pd.DataFrame) -> pd.Series:
    county_label = df["county_name"].fillna("") + " " + df["county_lsad"].fillna("")
    county_label = county_label.str.strip().replace("", "Unknown County")

    out = (
        "County: "
        + county_label
        + "<br>State FIPS: "
        + df["state_fips"].fillna("").astype(str)
        + "<br>County FIPS: "
        + df["FIPS"].astype(str)
        + "<br>Index for map: "
        + df["index_for_map"].round(2).astype(str)
        + "<br>Original index: "
        + df["index_original"].round(2).astype(str)
    )
    out[df["imputed"]] = out[df["imputed"]] + "<br>Status: imputed"
    out[~df["imputed"]] = out[~df["imputed"]] + "<br>Status: observed"
    return out


def _build_missing_hover_text(df: pd.DataFrame) -> pd.Series:
    county_label = df["county_name"].fillna("") + " " + df["county_lsad"].fillna("")
    county_label = county_label.str.strip().replace("", "Unknown County")
    return (
        "County: "
        + county_label
        + "<br>State FIPS: "
        + df["state_fips"].fillna("").astype(str)
        + "<br>County FIPS: "
        + df["FIPS"].astype(str)
        + "<br>Status: missing index"
    )


def create_county_map(config: MapConfig) -> Dict[str, Any]:
    geojson = _load_geojson(config.geojson_cache, config.geojson_url)
    county_meta = _extract_county_metadata(geojson)
    input_df, duplicate_fips = _prepare_index_data(
        path=config.index_file,
        fips_col=config.fips_col,
        index_col=config.index_col,
    )

    merged = county_meta.merge(input_df, on="FIPS", how="left")
    merged = _apply_missing_strategy(merged, config.missing_strategy)

    data_for_color = merged[merged["index_for_map"].notna()].copy()
    missing_for_map = merged[merged["missing_after_strategy"]].copy()
    zmin = float(data_for_color["index_for_map"].min()) if len(data_for_color) else 0.0
    zmax = float(data_for_color["index_for_map"].max()) if len(data_for_color) else 1.0

    fig = go.Figure()

    if config.engine == "geo":
        if len(data_for_color):
            fig.add_trace(
                go.Choropleth(
                    geojson=geojson,
                    featureidkey="id",
                    locations=data_for_color["FIPS"],
                    z=data_for_color["index_for_map"],
                    text=_build_hover_text(data_for_color),
                    hovertemplate="%{text}<extra></extra>",
                    colorscale="YlOrRd",
                    zmin=zmin,
                    zmax=zmax,
                    marker_line_color="white",
                    marker_line_width=0.2,
                    colorbar_title="Food Insecurity Index",
                    name="Index",
                )
            )

        if len(missing_for_map):
            fig.add_trace(
                go.Choropleth(
                    geojson=geojson,
                    featureidkey="id",
                    locations=missing_for_map["FIPS"],
                    z=np.ones(len(missing_for_map)),
                    text=_build_missing_hover_text(missing_for_map),
                    hovertemplate="%{text}<extra></extra>",
                    colorscale=[[0.0, "#BDBDBD"], [1.0, "#BDBDBD"]],
                    marker_line_color="white",
                    marker_line_width=0.2,
                    showscale=False,
                    name="Missing index",
                )
            )
        fig.update_geos(
            scope="usa",
            projection_type="albers usa",
            visible=True,
            showcoastlines=False,
            showcountries=False,
            showsubunits=False,
            showland=True,
            landcolor="rgb(245, 245, 245)",
            showlakes=True,
            lakecolor="rgb(229, 236, 246)",
            bgcolor="rgba(0,0,0,0)",
        )
    elif config.engine == "mapbox":
        if len(data_for_color):
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson,
                    featureidkey="id",
                    locations=data_for_color["FIPS"],
                    z=data_for_color["index_for_map"],
                    text=_build_hover_text(data_for_color),
                    hovertemplate="%{text}<extra></extra>",
                    colorscale="YlOrRd",
                    zmin=zmin,
                    zmax=zmax,
                    marker_line_color="white",
                    marker_line_width=0.2,
                    colorbar_title="Food Insecurity Index",
                    name="Index",
                )
            )

        if len(missing_for_map):
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=geojson,
                    featureidkey="id",
                    locations=missing_for_map["FIPS"],
                    z=np.ones(len(missing_for_map)),
                    text=_build_missing_hover_text(missing_for_map),
                    hovertemplate="%{text}<extra></extra>",
                    colorscale=[[0.0, "#BDBDBD"], [1.0, "#BDBDBD"]],
                    marker_line_color="white",
                    marker_line_width=0.2,
                    showscale=False,
                    name="Missing index",
                )
            )
    else:
        raise ValueError(f"Unsupported engine: {config.engine}")

    imputed_count = int(merged["imputed"].sum())
    missing_after = int(merged["missing_after_strategy"].sum())
    title = f"{config.title}<br><sup>{config.subtitle}</sup>"
    fig.update_layout(
        title=title,
        margin={"r": 0, "t": 80, "l": 0, "b": 0},
        template="plotly_white",
        legend_title_text="Legend",
    )
    if config.engine == "mapbox":
        fig.update_layout(
            mapbox={
                "style": "white-bg",
                "center": {"lat": 37.0902, "lon": -95.7129},
                "zoom": 3,
            }
        )

    config.output_html.parent.mkdir(parents=True, exist_ok=True)
    config.output_data.parent.mkdir(parents=True, exist_ok=True)
    config.output_missing.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(config.output_data, index=False)
    merged.loc[merged["missing_after_strategy"], ["FIPS", "county_name", "state_fips"]].to_csv(
        config.output_missing, index=False
    )
    # Self-contained HTML avoids blank maps in IDE previews/offline environments.
    fig.write_html(str(config.output_html), include_plotlyjs=True)

    summary = {
        "input_file": str(config.index_file),
        "engine": config.engine,
        "missing_strategy": config.missing_strategy,
        "total_counties_geojson": int(len(county_meta)),
        "input_unique_counties": int(len(input_df)),
        "duplicate_fips_in_input": duplicate_fips,
        "missing_original_count": int(merged["missing_original"].sum()),
        "imputed_count": imputed_count,
        "missing_after_strategy_count": missing_after,
        "output_html": str(config.output_html),
        "output_data": str(config.output_data),
        "output_missing": str(config.output_missing),
    }
    config.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
