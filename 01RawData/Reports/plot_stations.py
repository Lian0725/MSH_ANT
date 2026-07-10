#!/usr/bin/env python3
"""Use GMT to plot clean-frame Mount St. Helens station maps, revised."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_ROOT = Path(os.environ.get("MSH_ANT_METADATA_ROOT", PROJECT_ROOT / "data" / "metadata"))
OUTPUT_DIR = Path(
    os.environ.get(
        "MSH_ANT_OUTPUT_DIR",
        PROJECT_ROOT / "outputs" / "figures" / "station_maps",
    )
)
GMT = shutil.which("gmt") or "gmt"
Image.MAX_IMAGE_PIXELS = None

CRATER_LON = -122.1944
CRATER_LAT = 46.1912
VOLCANOES = (
    (-121.7603, 46.8523),
    (-121.4000, 46.5000),
    (-122.1944, 46.1912),
    (-121.4909, 46.2062),
    (-121.6959, 45.3735),
    (-121.7993, 44.6743),
)
DEFAULT_PROJECTION = "M16c"
DEFAULT_RELIEF = "@earth_relief_15s"
INSET_RELIEF = "@earth_relief_01s"
FOCUSED_REGION = (-123.1944, -121.1944, 45.5912, 46.7912)

SOFT_TERRAIN_CPT = """\
-1000  116/178/196   -1     218/239/236
-1     218/239/236   0      241/245/236
0      232/237/218   500    209/222/187
500    209/222/187   1500   203/192/156
1500   203/192/156   3000   226/214/183
3000   226/214/183   4500   239/230/204
B 116/178/196
F 239/230/204
N 235/235/235
"""

CANVAS_WIDTH = 2200
CANVAS_HEIGHT = 1900
TITLE_Y = 30
MAP_BOX = (80, 135, 2120, 1810)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
VOLCANO_FILL_HEX = "#d63b2f"
VOLCANO_FILL = (214, 59, 47)
PAPER_BG = (250, 248, 242)
INK = (28, 28, 26)
LIGHT_GRAY = (218, 215, 207)
MID_GRAY = (92, 89, 84)
LEGEND_BG = (255, 255, 255)


@dataclass(frozen=True)
class Station:
    code: str
    lon: float
    lat: float


@dataclass(frozen=True)
class NetworkStyle:
    label: str
    color: str
    symbol: str
    size_cm: float
    legend_shape: str
    rgb: tuple[int, int, int]
    pen: str = "0.45p,black"
    halo_cm: float = 0.05


@dataclass(frozen=True)
class MapConfig:
    name: str
    title: str
    output_png: Path
    networks: list[tuple[Path, NetworkStyle]]
    fixed_region: tuple[float, float, float, float] | None = None
    inset_focus_label: str | None = None
    inset_include_labels: tuple[str, ...] = ()
    inset_size_px: int = 620
    inset_anchor: str = "bottom-right"
    region_pad_lon: float = 0.0
    region_pad_lat: float = 0.0


@dataclass(frozen=True)
class RenderSpec:
    name: str
    region: tuple[float, float, float, float]
    projection: str
    relief_grid: str
    show_axes: bool
    show_crater_label: bool
    output_png: Path
    show_crater_marker: bool = True
    dpi: int = 260
    crater_size_cm: float = 0.48


@dataclass(frozen=True)
class PlacedImage:
    left: int
    top: int
    width: int
    height: int


def metadata_dir(year: str, network: str) -> Path:
    """Resolve both project and organized RawData metadata layouts."""
    candidates = (
        METADATA_ROOT / year / network,
        METADATA_ROOT / year / "MetaData" / network,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_stationxml(xml_path: Path) -> list[Station]:
    ns = {"fdsn": "http://www.fdsn.org/xml/station/1"}
    root = ET.parse(xml_path).getroot()
    stations: list[Station] = []

    network = root.find(".//fdsn:Network", ns)
    network_code = network.attrib.get("code", "") if network is not None else ""

    for sta in root.findall(".//fdsn:Station", ns):
        code = sta.attrib.get("code")
        lat = sta.find("fdsn:Latitude", ns)
        lon = sta.find("fdsn:Longitude", ns)
        if code and lat is not None and lon is not None:
            full_code = f"{network_code}.{code}" if network_code else code
            stations.append(Station(code=full_code, lon=float(lon.text), lat=float(lat.text)))

    if stations:
        return stations

    for sta in root.iter():
        if not sta.tag.endswith("Station"):
            continue
        code = sta.attrib.get("code")
        lat = None
        lon = None
        for child in sta:
            if child.tag.endswith("Latitude"):
                lat = float(child.text)
            elif child.tag.endswith("Longitude"):
                lon = float(child.text)
        if code and lat is not None and lon is not None:
            full_code = f"{network_code}.{code}" if network_code else code
            stations.append(Station(code=full_code, lon=lon, lat=lat))
    return stations


def load_stations(xml_dir: Path) -> list[Station]:
    dedup: dict[str, Station] = {}
    for xml_path in sorted(xml_dir.rglob("*.xml")):
        for station in parse_stationxml(xml_path):
            dedup[station.code] = station
    if not dedup:
        raise RuntimeError(f"No station coordinates found in {xml_dir}")
    return sorted(dedup.values(), key=lambda item: item.code)


def mercator_y(lat_deg: float) -> float:
    lat_rad = math.radians(lat_deg)
    return math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))


def centered_region(
    all_points: Iterable[tuple[float, float]],
    target_aspect: float = 1.55,
) -> tuple[float, float, float, float]:
    points = list(all_points)
    if not points:
        raise ValueError("No points for region calculation")

    max_lon_delta = max(abs(lon - CRATER_LON) for lon, _ in points) + 0.08
    max_lat_delta = max(abs(lat - CRATER_LAT) for _, lat in points) + 0.06

    span_lon = max(0.28, min(max_lon_delta, 1.75))
    span_lat = max(0.22, min(max_lat_delta, 1.05))

    current_aspect = (span_lon * 2.0) / (span_lat * 2.0)
    if current_aspect < target_aspect:
        span_lon = min(span_lat * target_aspect, 1.85)
    else:
        span_lat = min(span_lon / target_aspect, 1.15)

    return (
        CRATER_LON - span_lon,
        CRATER_LON + span_lon,
        CRATER_LAT - span_lat,
        CRATER_LAT + span_lat,
    )


def zoom_region(stations: Iterable[Station]) -> tuple[float, float, float, float]:
    pts = list(stations)
    lons = [station.lon for station in pts]
    lats = [station.lat for station in pts]
    west = min(lons) - 0.015
    east = max(lons) + 0.015
    south = min(lats) - 0.015
    north = max(lats) + 0.015

    lon_span = east - west
    lat_span = north - south
    target_aspect = 1.28
    current_aspect = lon_span / lat_span if lat_span else target_aspect
    if current_aspect < target_aspect:
        half_lon = lat_span * target_aspect / 2.0
        center_lon = (west + east) / 2.0
        west = center_lon - half_lon
        east = center_lon + half_lon
    else:
        half_lat = lon_span / target_aspect / 2.0
        center_lat = (south + north) / 2.0
        south = center_lat - half_lat
        north = center_lat + half_lat
    return west, east, south, north


def expand_region(
    region: tuple[float, float, float, float],
    pad_lon: float = 0.0,
    pad_lat: float = 0.0,
) -> tuple[float, float, float, float]:
    west, east, south, north = region
    return west - pad_lon, east + pad_lon, south - pad_lat, north + pad_lat


def write_xy(path: Path, stations: Iterable[Station]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for station in stations:
            handle.write(f"{station.lon:.6f} {station.lat:.6f}\n")


def write_single_point(path: Path, lon: float, lat: float) -> None:
    path.write_text(f"{lon:.6f} {lat:.6f}\n", encoding="utf-8")


def write_points(path: Path, points: Iterable[tuple[float, float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for lon, lat in points:
            handle.write(f"{lon:.6f} {lat:.6f}\n")


def format_region(region: tuple[float, float, float, float]) -> str:
    west, east, south, north = region
    return f"{west:.6f}/{east:.6f}/{south:.6f}/{north:.6f}"


def frame_intervals(region: tuple[float, float, float, float]) -> tuple[str, str]:
    west, east, south, north = region
    lon_span = east - west
    lat_span = north - south

    if lon_span > 2.4:
        x_major = "1"
    elif lon_span > 1.2:
        x_major = "30m"
    else:
        x_major = "15m"

    if lat_span > 1.0:
        y_major = "30m"
    elif lat_span > 0.5:
        y_major = "20m"
    else:
        y_major = "10m"

    return f"-Bxa{x_major}", f"-Bya{y_major}"


def run_shell_script(script: str, workdir: Path) -> None:
    subprocess.run(
        ["/bin/bash", "-lc", script],
        cwd=workdir,
        check=True,
        text=True,
    )


def build_gmt_script(
    render: RenderSpec,
    volcano_file: str,
    network_files: list[tuple[str, NetworkStyle]],
) -> str:
    region_text = format_region(render.region)
    west, east, south, north = render.region
    x_frame, y_frame = frame_intervals(render.region)
    plot_lines = []
    keep_open_after_networks = render.show_crater_marker
    station_commands = []
    for data_file, style in network_files:
        halo_size = style.size_cm + style.halo_cm
        station_commands.append(
            f'{GMT} psxy {data_file} -R{region_text} -J{render.projection} '
            f'-S{style.symbol}{halo_size:.2f}c -Gwhite -W0.20p,white'
        )
        station_commands.append(
            f'{GMT} psxy {data_file} -R{region_text} -J{render.projection} '
            f'-S{style.symbol}{style.size_cm:.2f}c -G{style.color} -W{style.pen}'
        )
    for idx, command in enumerate(station_commands):
        is_last = idx == len(station_commands) - 1
        close_flag = "-O -K" if (not is_last or keep_open_after_networks) else "-O"
        plot_lines.append(f"{command} {close_flag} >> {render.name}.ps")

    if render.show_axes:
        basemap_line = (
            f'{GMT} psbasemap -R{region_text} -J{render.projection} {x_frame} {y_frame} '
            f'-BWS -O -K >> {render.name}.ps\n'
            f"cat <<'EOF' > frame_ne.xy\n"
            f">\n{west:.6f} {north:.6f}\n{east:.6f} {north:.6f}\n"
            f">\n{east:.6f} {south:.6f}\n{east:.6f} {north:.6f}\n"
            f"EOF\n"
            f'{GMT} psxy frame_ne.xy -R{region_text} -J{render.projection} '
            f'-W1.10p,black -O -K >> {render.name}.ps'
        )
        map_frame_type = "plain"
    else:
        basemap_line = f'{GMT} psbasemap -R{region_text} -J{render.projection} -B0 -O -K >> {render.name}.ps'
        map_frame_type = "plain"

    crater_plot = ""
    crater_label = ""
    if render.show_crater_marker:
        crater_plot = (
            f"{GMT} psxy {volcano_file} -R{region_text} -J{render.projection} "
            f"-St{render.crater_size_cm:.2f}c -G{VOLCANO_FILL_HEX} -W0.65p,black "
        )
        if render.show_crater_label:
            crater_plot += f"-O -K >> {render.name}.ps"
            crater_label = f"""cat <<'EOF' > crater_label.txt
{CRATER_LON + 0.018:.6f} {CRATER_LAT + 0.018:.6f} St. Helens crater
EOF
{GMT} pstext crater_label.txt -R{region_text} -J{render.projection} -F+f8p,Helvetica-Bold,red+jLB -D0.04c/0.02c -O >> {render.name}.ps
"""
        else:
            crater_plot += f"-O >> {render.name}.ps"

    return f"""
set -euo pipefail
{GMT} set MAP_FRAME_TYPE {map_frame_type} \
    FORMAT_GEO_MAP ddd:mmF \
    MAP_GRID_PEN_PRIMARY 0.10p,gray82 \
    MAP_TICK_PEN_PRIMARY 0.55p,black \
    MAP_FRAME_PEN 1.10p,black \
    MAP_TICK_LENGTH_PRIMARY 4p \
    MAP_ANNOT_OFFSET_PRIMARY 2p \
    FONT_ANNOT_PRIMARY 8p,Helvetica,black \
    FONT_LABEL 9p,Helvetica,black
{GMT} grdcut {render.relief_grid} -R{region_text} -Grelief.nc
{GMT} grdgradient relief.nc -A315 -Ne0.38 -Gshade.nc
cat <<'EOF' > relief.cpt
{SOFT_TERRAIN_CPT.rstrip()}
EOF
{GMT} grdimage relief.nc -R{region_text} -J{render.projection} -Crelief.cpt -Ishade.nc -K -P > {render.name}.ps
{GMT} pscoast -R{region_text} -J{render.projection} -Df -W0.18p,gray65 -N1/0.12p,gray70 -O -K >> {render.name}.ps
{basemap_line}
{chr(10).join(plot_lines)}
{crater_plot}
{crater_label}{GMT} psconvert {render.name}.ps -Tg -A+m0.04c -P -E{render.dpi} -F{render.name}
"""


def render_gmt_image(
    render: RenderSpec,
    volcano_file: Path,
    network_files: list[tuple[str, NetworkStyle]],
    workdir: Path,
) -> Path:
    script = build_gmt_script(render, volcano_file.name, network_files)
    run_shell_script(script, workdir)
    if not render.output_png.exists():
        raise RuntimeError(f"GMT did not produce {render.output_png.name}")
    return render.output_png


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                "/System/Library/Fonts/Supplemental/Helvetica.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        )
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def fit_into_box(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[Image.Image, PlacedImage]:
    left, top, right, bottom = box
    max_width = right - left
    max_height = bottom - top
    scale = min(max_width / image.width, max_height / image.height)
    width = int(round(image.width * scale))
    height = int(round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.LANCZOS)
    placed = PlacedImage(
        left=left + (max_width - width) // 2,
        top=top + (max_height - height) // 2,
        width=width,
        height=height,
    )
    return resized, placed


def detect_map_frame(image: Image.Image) -> tuple[int, int, int, int]:
    gray = image.convert("L")
    width, height = image.size
    dark_threshold = 235

    top = 0
    for y in range(height):
        row_dark = sum(1 for x in range(width) if gray.getpixel((x, y)) < dark_threshold)
        if row_dark > width * 0.45:
            top = y
            break

    bottom = height - 1
    for y in range(height - 1, -1, -1):
        row_dark = sum(1 for x in range(width) if gray.getpixel((x, y)) < dark_threshold)
        if row_dark > width * 0.45:
            bottom = y
            break

    left = 0
    for x in range(width):
        col_dark = sum(1 for y in range(height) if gray.getpixel((x, y)) < dark_threshold)
        if col_dark > height * 0.45:
            left = x
            break

    right = width - 1
    for x in range(width - 1, -1, -1):
        col_dark = sum(1 for y in range(height) if gray.getpixel((x, y)) < dark_threshold)
        if col_dark > height * 0.45:
            right = x
            break

    inset = 12
    return left + inset, top + inset, right - inset, bottom - inset


def project_point(
    lon: float,
    lat: float,
    region: tuple[float, float, float, float],
    inner_box: tuple[int, int, int, int],
) -> tuple[float, float]:
    west, east, south, north = region
    left, top, right, bottom = inner_box

    x_frac = (lon - west) / (east - west)
    north_m = mercator_y(north)
    south_m = mercator_y(south)
    lat_m = mercator_y(lat)
    y_frac = (north_m - lat_m) / (north_m - south_m)

    x = left + x_frac * (right - left)
    y = top + y_frac * (bottom - top)
    return x, y


def scale_box(
    box: tuple[float, float, float, float],
    original_size: tuple[int, int],
    placed: PlacedImage,
) -> tuple[int, int, int, int]:
    src_w, src_h = original_size
    x_scale = placed.width / src_w
    y_scale = placed.height / src_h
    left, top, right, bottom = box
    return (
        int(round(placed.left + left * x_scale)),
        int(round(placed.top + top * y_scale)),
        int(round(placed.left + right * x_scale)),
        int(round(placed.top + bottom * y_scale)),
    )


def draw_legend_symbol(draw: ImageDraw.ImageDraw, x: int, y: int, style: NetworkStyle) -> None:
    if style.legend_shape == "circle":
        r = 12
        draw.ellipse((x - r - 4, y - r - 4, x + r + 4, y + r + 4), fill=WHITE)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=style.rgb, outline=BLACK, width=3)
    elif style.legend_shape == "triangle":
        halo = [(x, y - 18), (x - 17, y + 15), (x + 17, y + 15)]
        draw.polygon(halo, fill=WHITE)
        pts = [(x, y - 14), (x - 13, y + 12), (x + 13, y + 12)]
        draw.polygon(pts, fill=style.rgb, outline=BLACK)
    elif style.legend_shape == "inverted_triangle":
        halo = [(x, y + 18), (x - 17, y - 15), (x + 17, y - 15)]
        draw.polygon(halo, fill=WHITE)
        pts = [(x, y + 14), (x - 13, y - 12), (x + 13, y - 12)]
        draw.polygon(pts, fill=style.rgb, outline=BLACK)
    elif style.legend_shape == "square":
        r = 12
        draw.rectangle((x - r - 4, y - r - 4, x + r + 4, y + r + 4), fill=WHITE)
        draw.rectangle((x - r, y - r, x + r, y + r), fill=style.rgb, outline=BLACK, width=3)


def draw_map_legend(
    draw: ImageDraw.ImageDraw,
    title_font: ImageFont.ImageFont,
    label_font: ImageFont.ImageFont,
    network_styles: list[NetworkStyle],
    anchor: tuple[int, int],
) -> None:
    title = "Station networks"
    padding_x = 20
    padding_y = 16
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    item_bboxes = [draw.textbbox((0, 0), style.label, font=label_font) for style in network_styles]
    panel_width = max(
        title_bbox[2] - title_bbox[0],
        max((bbox[2] - bbox[0] for bbox in item_bboxes), default=0) + 62,
    ) + padding_x * 2
    row_h = 42
    panel_height = padding_y * 2 + 34 + row_h * len(network_styles)
    x0, y0 = anchor
    panel = (x0, y0, x0 + panel_width, y0 + panel_height)
    shadow = (panel[0] + 6, panel[1] + 7, panel[2] + 6, panel[3] + 7)

    draw.rounded_rectangle(shadow, radius=12, fill=(55, 55, 55))
    draw.rounded_rectangle(panel, radius=12, fill=LEGEND_BG, outline=(28, 28, 28), width=2)
    draw.text((x0 + padding_x, y0 + padding_y - 2), title, fill=INK, font=title_font)

    row_y = y0 + padding_y + 48
    symbol_x = x0 + padding_x + 16
    text_x = x0 + padding_x + 50
    for style in network_styles:
        draw_legend_symbol(draw, symbol_x, row_y + 12, style)
        draw.text((text_x, row_y - 4), style.label, fill=INK, font=label_font)
        row_y += row_h


def draw_scale_bar(
    draw: ImageDraw.ImageDraw,
    region: tuple[float, float, float, float],
    inner_box: tuple[int, int, int, int],
    anchor_side: str,
) -> None:
    west, east, south, north = region
    left, top, right, bottom = inner_box
    center_lat = (south + north) / 2.0
    lon_span_for_30km = 30.0 / (111.32 * math.cos(math.radians(center_lat)))
    pixels_per_lon = (right - left) / (east - west)
    bar_width = int(round(lon_span_for_30km * pixels_per_lon))

    margin_x = 86
    y = bottom - 82
    tick = 30
    if anchor_side == "right":
        x0 = right - margin_x - bar_width
    else:
        x0 = left + margin_x
    x1 = x0 + bar_width

    label_font = load_font(28, bold=True)
    label = "30KM"
    label_box = draw.textbbox(((x0 + x1) / 2, y - tick - 12), label, font=label_font, anchor="ms")

    # White under-stroke keeps the scale readable without adding a panel.
    draw.line((x0, y, x1, y), fill=WHITE, width=9)
    draw.line((x0, y, x0, y - tick), fill=WHITE, width=9)
    draw.line((x1, y, x1, y - tick), fill=WHITE, width=9)
    draw.text(
        ((x0 + x1) / 2, y - tick - 12),
        label,
        fill=WHITE,
        font=label_font,
        anchor="ms",
        stroke_width=4,
        stroke_fill=WHITE,
    )

    draw.line((x0, y, x1, y), fill=BLACK, width=4)
    draw.line((x0, y, x0, y - tick), fill=BLACK, width=4)
    draw.line((x1, y, x1, y - tick), fill=BLACK, width=4)
    draw.text(
        ((label_box[0] + label_box[2]) / 2, y - tick - 12),
        label,
        fill=BLACK,
        font=label_font,
        anchor="ms",
    )


def draw_volcano_markers(
    draw: ImageDraw.ImageDraw,
    region: tuple[float, float, float, float],
    inner_box: tuple[int, int, int, int],
    size_px: int = 26,
) -> None:
    west, east, south, north = region
    for lon, lat in VOLCANOES:
        if not (west <= lon <= east and south <= lat <= north):
            continue
        x, y = project_point(lon, lat, region, inner_box)
        halo_size = size_px + 4
        halo = [
            (x, y - halo_size),
            (x - halo_size * 0.90, y + halo_size * 0.78),
            (x + halo_size * 0.90, y + halo_size * 0.78),
        ]
        pts = [
            (x, y - size_px),
            (x - size_px * 0.90, y + size_px * 0.78),
            (x + size_px * 0.90, y + size_px * 0.78),
        ]
        draw.polygon(halo, fill=WHITE)
        draw.polygon(pts, fill=VOLCANO_FILL)
        draw.line((*pts, pts[0]), fill=BLACK, width=3)


def compose_canvas(
    config: MapConfig,
    main_raw: Path,
    network_styles: list[NetworkStyle],
    region: tuple[float, float, float, float],
    inset_focus_stations: list[Station] | None = None,
    inset_raw: Path | None = None,
) -> None:
    main_image = Image.open(main_raw).convert("RGB")
    canvas = main_image.copy()
    draw = ImageDraw.Draw(canvas)
    legend_title_font = load_font(24, bold=True)
    legend_label_font = load_font(27, bold=False)

    inner_main = detect_map_frame(main_image)
    draw_map_legend(
        draw,
        legend_title_font,
        legend_label_font,
        network_styles,
        anchor=(inner_main[0] + 34, inner_main[1] + 34),
    )
    scale_anchor_side = "right" if config.inset_anchor == "bottom-left" else "left"
    draw_scale_bar(draw, region, inner_main, scale_anchor_side)

    if inset_focus_stations and inset_raw:
        inset_region = zoom_region(inset_focus_stations)
        west, east, south, north = inset_region
        box_pad_lon = 0.002
        box_pad_lat = 0.002

        tl = project_point(west - box_pad_lon, north + box_pad_lat, region, inner_main)
        br = project_point(east + box_pad_lon, south - box_pad_lat, region, inner_main)
        highlight_box = (
            int(round(tl[0])),
            int(round(tl[1])),
            int(round(br[0])),
            int(round(br[1])),
        )
        draw.rectangle(highlight_box, outline=WHITE, width=9)
        draw.rectangle(highlight_box, outline=(36, 36, 34), width=4)

        inset_image = Image.open(inset_raw).convert("RGB")
        inset_margin = 38
        inset_size = config.inset_size_px
        if config.inset_anchor == "bottom-left":
            inset_target = (
                inner_main[0] + inset_margin,
                inner_main[3] - inset_size - inset_margin,
                inner_main[0] + inset_size + inset_margin,
                inner_main[3] - inset_margin,
            )
        else:
            inset_target = (
                inner_main[2] - inset_size - inset_margin,
                inner_main[3] - inset_size - inset_margin,
                inner_main[2] - inset_margin,
                inner_main[3] - inset_margin,
            )
        inset_scaled, inset_placed = fit_into_box(inset_image, inset_target)
        inset_mat = (
            inset_placed.left - 10,
            inset_placed.top - 10,
            inset_placed.left + inset_placed.width + 10,
            inset_placed.top + inset_placed.height + 10,
        )
        draw.rectangle(inset_mat, fill=WHITE)
        canvas.paste(inset_scaled, (inset_placed.left, inset_placed.top))
        draw.rectangle(inset_mat, outline=(55, 55, 52), width=3)

        if config.inset_anchor == "bottom-left":
            guide_start_top = (highlight_box[0], highlight_box[1])
            guide_start_bottom = (highlight_box[0], highlight_box[3])
            guide_end_top = (inset_mat[2], inset_mat[1])
            guide_end_bottom = (inset_mat[2], inset_mat[3])
        else:
            guide_start_top = (highlight_box[2], highlight_box[1])
            guide_start_bottom = (highlight_box[2], highlight_box[3])
            guide_end_top = (inset_mat[0], inset_mat[1])
            guide_end_bottom = (inset_mat[0], inset_mat[3])
        draw.line((guide_start_top, guide_end_top), fill=WHITE, width=7)
        draw.line((guide_start_bottom, guide_end_bottom), fill=WHITE, width=7)
        draw.line((guide_start_top, guide_end_top), fill=(42, 42, 40), width=3)
        draw.line((guide_start_bottom, guide_end_bottom), fill=(42, 42, 40), width=3)

    canvas.save(config.output_png)
    canvas.save(config.output_png.with_suffix(".pdf"), "PDF", resolution=300.0)
    return

    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), PAPER_BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(54, bold=True)
    legend_title_font = load_font(24, bold=True)
    legend_label_font = load_font(27, bold=False)

    title_w = draw.textbbox((0, 0), config.title, font=title_font)[2]
    draw.text(((CANVAS_WIDTH - title_w) / 2, TITLE_Y), config.title, fill=INK, font=title_font)

    main_image = Image.open(main_raw).convert("RGB")
    main_scaled, main_placed = fit_into_box(main_image, MAP_BOX)
    main_mat = (
        main_placed.left - 18,
        main_placed.top - 18,
        main_placed.left + main_placed.width + 18,
        main_placed.top + main_placed.height + 18,
    )
    main_shadow = (main_mat[0] + 10, main_mat[1] + 12, main_mat[2] + 10, main_mat[3] + 12)
    draw.rectangle(main_shadow, fill=(228, 225, 217))
    draw.rectangle(main_mat, fill=WHITE)
    canvas.paste(main_scaled, (main_placed.left, main_placed.top))

    inner_main = detect_map_frame(main_image)
    inner_main_canvas = scale_box(inner_main, main_image.size, main_placed)
    draw_map_legend(
        draw,
        legend_title_font,
        legend_label_font,
        network_styles,
        anchor=(inner_main_canvas[0] + 34, inner_main_canvas[1] + 34),
    )
    scale_anchor_side = "right" if config.inset_anchor == "bottom-left" else "left"
    draw_scale_bar(draw, region, inner_main_canvas, scale_anchor_side)

    if inset_focus_stations and inset_raw:
        inset_region = zoom_region(inset_focus_stations)
        west, east, south, north = inset_region
        box_pad_lon = 0.002
        box_pad_lat = 0.002

        tl = project_point(west - box_pad_lon, north + box_pad_lat, region, inner_main)
        br = project_point(east + box_pad_lon, south - box_pad_lat, region, inner_main)
        highlight_box = scale_box((tl[0], tl[1], br[0], br[1]), main_image.size, main_placed)
        draw.rectangle(highlight_box, outline=WHITE, width=9)
        draw.rectangle(highlight_box, outline=(36, 36, 34), width=4)

        inset_image = Image.open(inset_raw).convert("RGB")
        inset_margin = 38
        inset_size = config.inset_size_px
        if config.inset_anchor == "bottom-left":
            inset_target = (
                inner_main_canvas[0] + inset_margin,
                inner_main_canvas[3] - inset_size - inset_margin,
                inner_main_canvas[0] + inset_size + inset_margin,
                inner_main_canvas[3] - inset_margin,
            )
        else:
            inset_target = (
                inner_main_canvas[2] - inset_size - inset_margin,
                inner_main_canvas[3] - inset_size - inset_margin,
                inner_main_canvas[2] - inset_margin,
                inner_main_canvas[3] - inset_margin,
            )
        inset_scaled, inset_placed = fit_into_box(inset_image, inset_target)
        inset_mat = (
            inset_placed.left - 10,
            inset_placed.top - 10,
            inset_placed.left + inset_placed.width + 10,
            inset_placed.top + inset_placed.height + 10,
        )
        inset_shadow = (inset_mat[0] + 8, inset_mat[1] + 10, inset_mat[2] + 8, inset_mat[3] + 10)
        draw.rectangle(inset_shadow, fill=(210, 207, 199))
        draw.rectangle(inset_mat, fill=WHITE)
        canvas.paste(inset_scaled, (inset_placed.left, inset_placed.top))
        draw.rectangle(
            inset_mat,
            outline=(55, 55, 52),
            width=3,
        )

        if config.inset_anchor == "bottom-left":
            guide_start_top = (highlight_box[0], highlight_box[1])
            guide_start_bottom = (highlight_box[0], highlight_box[3])
            guide_end_top = (inset_mat[2], inset_mat[1])
            guide_end_bottom = (inset_mat[2], inset_mat[3])
        else:
            guide_start_top = (highlight_box[2], highlight_box[1])
            guide_start_bottom = (highlight_box[2], highlight_box[3])
            guide_end_top = (inset_mat[0], inset_mat[1])
            guide_end_bottom = (inset_mat[0], inset_mat[3])
        draw.line((guide_start_top, guide_end_top), fill=WHITE, width=7)
        draw.line((guide_start_bottom, guide_end_bottom), fill=WHITE, width=7)
        draw.line((guide_start_top, guide_end_top), fill=(42, 42, 40), width=3)
        draw.line((guide_start_bottom, guide_end_bottom), fill=(42, 42, 40), width=3)

    canvas.save(config.output_png)


def generate_map(config: MapConfig) -> dict[str, object]:
    config.output_png.parent.mkdir(parents=True, exist_ok=True)
    all_points: list[tuple[float, float]] = [(CRATER_LON, CRATER_LAT)]
    summary_networks: list[dict[str, object]] = []
    network_files: list[tuple[str, NetworkStyle]] = []
    network_styles: list[NetworkStyle] = []
    loaded_networks: dict[str, list[Station]] = {}

    with tempfile.TemporaryDirectory(prefix=f"{config.name}_gmt_") as tmpdir:
        tmp = Path(tmpdir)
        for xml_dir, style in config.networks:
            stations = load_stations(xml_dir)
            loaded_networks[style.label] = stations
            summary_networks.append(
                {
                    "label": style.label,
                    "count": len(stations),
                    "source_dir": str(xml_dir),
                }
            )
            all_points.extend((station.lon, station.lat) for station in stations)
            xy_path = tmp / f"{style.label.lower().replace(' ', '_')}.xy"
            write_xy(xy_path, stations)
            network_files.append((xy_path.name, style))
            network_styles.append(style)

        if config.fixed_region is not None:
            region = config.fixed_region
        else:
            region = expand_region(
                centered_region(all_points),
                pad_lon=config.region_pad_lon,
                pad_lat=config.region_pad_lat,
            )
        volcano_file = tmp / "volcanoes.xy"
        write_points(volcano_file, VOLCANOES)

        main_render = RenderSpec(
            name=f"{config.name}_raw",
            region=region,
            projection=DEFAULT_PROJECTION,
            relief_grid=DEFAULT_RELIEF,
            show_axes=True,
            show_crater_label=False,
            output_png=tmp / f"{config.name}_raw.png",
        )
        main_raw = render_gmt_image(main_render, volcano_file, network_files, tmp)

        inset_raw = None
        inset_focus_stations = None
        if config.inset_focus_label:
            inset_focus_stations = loaded_networks[config.inset_focus_label]
            inset_region = zoom_region(inset_focus_stations)
            inset_network_files: list[tuple[str, NetworkStyle]] = []
            for filename, style in network_files:
                if config.inset_include_labels and style.label not in config.inset_include_labels:
                    continue
                if style.label in {"YI", "1D"}:
                    inset_size_cm = 0.24 if style.label == "1D" else 0.44
                    inset_halo_cm = 0.06 if style.label == "1D" else 0.10
                    if style.label == "YI" and config.name == "maps_2017":
                        inset_size_cm = 0.26
                        inset_halo_cm = 0.06
                    inset_style = NetworkStyle(
                        style.label,
                        style.color,
                        style.symbol,
                        inset_size_cm,
                        style.legend_shape,
                        style.rgb,
                        style.pen,
                        inset_halo_cm,
                    )
                elif style.label == "Permanent":
                    inset_style = NetworkStyle(
                        style.label,
                        style.color,
                        style.symbol,
                        0.38,
                        style.legend_shape,
                        style.rgb,
                        style.pen,
                        0.08,
                    )
                elif style.label == "XD":
                    inset_style = NetworkStyle(
                        style.label,
                        style.color,
                        style.symbol,
                        0.50,
                        style.legend_shape,
                        style.rgb,
                        style.pen,
                        0.10,
                    )
                else:
                    inset_style = style
                inset_network_files.append((filename, inset_style))
            inset_render = RenderSpec(
                name=f"{config.name}_inset",
                region=inset_region,
                projection=DEFAULT_PROJECTION,
                relief_grid=INSET_RELIEF,
                show_axes=False,
                show_crater_label=False,
                show_crater_marker=False,
                output_png=tmp / f"{config.name}_inset.png",
                dpi=1400,
                crater_size_cm=0.82,
            )
            inset_raw = render_gmt_image(inset_render, volcano_file, inset_network_files, tmp)

        compose_canvas(
            config=config,
            main_raw=main_raw,
            network_styles=network_styles,
            region=region,
            inset_focus_stations=inset_focus_stations,
            inset_raw=inset_raw,
        )

    return {
        "name": config.name,
        "title": config.title,
        "output_png": str(config.output_png),
        "output_pdf": str(config.output_png.with_suffix(".pdf")),
        "projection": DEFAULT_PROJECTION,
        "relief_grid": DEFAULT_RELIEF,
        "inset_relief_grid": INSET_RELIEF if config.inset_focus_label else None,
        "crater_lon": CRATER_LON,
        "crater_lat": CRATER_LAT,
        "networks": summary_networks,
    }


def main() -> None:
    configs = [
        MapConfig(
            name="maps_2014",
            title="",
            output_png=OUTPUT_DIR / "maps_2014.png",
            fixed_region=FOCUSED_REGION,
            networks=[
                (
                    metadata_dir("2014", "1D"),
                    NetworkStyle("1D", "118/60/172", "c", 0.13, "circle", (118, 60, 172), "0.34p,black", 0.04),
                ),
                (
                    metadata_dir("2014", "XD"),
                    NetworkStyle("XD", "37/99/235", "i", 0.35, "inverted_triangle", (37, 99, 235), "0.52p,black", 0.08),
                ),
                (
                    metadata_dir("2014", "permanent"),
                    NetworkStyle("Permanent", "0/188/178", "s", 0.24, "square", (0, 188, 178), "0.46p,black", 0.06),
                ),
            ],
            inset_focus_label="1D",
            inset_include_labels=("1D", "XD", "Permanent"),
            inset_size_px=560,
            inset_anchor="bottom-left",
            region_pad_lon=0.18,
            region_pad_lat=0.11,
        ),
        MapConfig(
            name="maps_2017",
            title="",
            output_png=OUTPUT_DIR / "maps_2017.png",
            fixed_region=FOCUSED_REGION,
            networks=[
                (
                    metadata_dir("2017", "YI"),
                    NetworkStyle("YI", "37/99/235", "c", 0.26, "circle", (37, 99, 235), "0.46p,black", 0.07),
                ),
                (
                    metadata_dir("2017", "permanent"),
                    NetworkStyle("Permanent", "0/188/178", "s", 0.22, "square", (0, 188, 178), "0.44p,black", 0.06),
                ),
            ],
            inset_focus_label="YI",
            inset_include_labels=("YI", "Permanent"),
            inset_size_px=620,
            inset_anchor="bottom-right",
        ),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [generate_map(config) for config in configs]
    summary_path = OUTPUT_DIR / "maps_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
