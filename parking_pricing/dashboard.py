"""Interactive Bokeh/Panel dashboard.

    parking-pricing dashboard            # http://localhost:5006
    panel serve parking_pricing/dashboard.py

Pick a lot and a day, then press play: the dashboard replays the day reading
by reading, showing how each model's price reacts, where every lot's price
stands on the map, and which full lots are rerouting drivers.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import (
    Arrow,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    NumeralTickFormatter,
    Span,
    VeeHead,
)
from bokeh.plotting import figure

# absolute imports so `panel serve parking_pricing/dashboard.py` also works
from parking_pricing.config import DEFAULT_CONFIG, PricingConfig
from parking_pricing.data import load_dataset, lot_registry
from parking_pricing.geo import build_neighbor_map, haversine_km
from parking_pricing.replay import run_replay

# Categorical slots 1-3 of the validated reference palette (one per model),
# a recessive ink for annotations, and a single-hue blue ramp for magnitude.
SERIES = {
    "price_baseline": ("Model 1 - baseline", "#2a78d6"),
    "price_demand": ("Model 2 - demand", "#eb6834"),
    "price_competitive": ("Model 3 - competitive", "#1baf7a"),
}
INK_MUTED = "#8a8984"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"
# steps 250-700 of the blue ramp: the lightest steps would vanish into the surface
PRICE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
OCCUPANCY_COLOR = "#2a78d6"
REROUTE_COLOR = "#e34948"

pn.extension("tabulator", sizing_mode="stretch_width")


_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}


def _load(cfg: PricingConfig = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price the dataset once per config and share it across browser sessions."""
    key = json.dumps(cfg.to_dict(), sort_keys=True)
    if key not in _CACHE:
        _CACHE[key] = _price_dataset(cfg)
    return _CACHE[key]


def _price_dataset(cfg: PricingConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset()
    priced = run_replay(df, cfg)
    priced["day"] = priced["timestamp"].dt.strftime("%Y-%m-%d")
    priced["time"] = priced["timestamp"].dt.strftime("%H:%M")
    return priced, lot_registry(df)


def _display_positions(lots: pd.DataFrame, cluster_km: float = 0.05,
                       ring_deg: float = 0.002) -> pd.DataFrame:
    """Spread co-located lots on a small ring so every lot is visible on the map.

    Several lots in the data sit within a few metres of each other (same
    building); plotted at their true position they would hide each other.
    """
    pos = lots[["latitude", "longitude"]].copy()
    clusters: list[list[str]] = []
    for lot, row in pos.iterrows():
        for members in clusters:
            head = pos.loc[members[0]]
            if haversine_km(row.latitude, row.longitude, head.latitude, head.longitude) <= cluster_km:
                members.append(lot)
                break
        else:
            clusters.append([lot])
    for members in clusters:
        if len(members) < 2:
            continue
        lat0 = pos.loc[members, "latitude"].mean()
        lon0 = pos.loc[members, "longitude"].mean()
        for i, lot in enumerate(sorted(members)):
            angle = 2 * np.pi * i / len(members) + np.pi / 2
            pos.loc[lot, "latitude"] = lat0 + ring_deg * np.sin(angle)
            pos.loc[lot, "longitude"] = lon0 + ring_deg * np.cos(angle)
    return pos


def _legend_above(fig) -> None:
    legend = fig.legend[0]
    legend.orientation = "horizontal"
    legend.border_line_color = None
    legend.background_fill_alpha = 0
    legend.click_policy = "hide"
    legend.label_text_color = INK_SECONDARY
    fig.add_layout(legend, "above")


def _style(fig) -> None:
    fig.toolbar.logo = None
    fig.outline_line_color = None
    fig.background_fill_color = None
    fig.border_fill_color = None
    for axis in (fig.xaxis, fig.yaxis):
        axis.axis_line_color = GRID
        axis.major_tick_line_color = GRID
        axis.minor_tick_line_color = None
        axis.major_label_text_color = INK_SECONDARY
        axis.axis_label_text_color = INK_SECONDARY
        axis.axis_label_text_font_style = "normal"
    fig.xgrid.grid_line_color = None
    fig.ygrid.grid_line_color = GRID
    fig.title.text_color = "#0b0b0b"
    fig.title.text_font_style = "normal"
    fig.title.text_font_size = "13pt"


def create_app(cfg: PricingConfig = DEFAULT_CONFIG):
    priced, lots = _load(cfg)
    neighbors = build_neighbor_map(lots, cfg.neighbor_radius_km)
    days = sorted(priced["day"].unique())
    lot_ids = sorted(priced["lot_id"].unique())
    valid_lots = lots[lots["valid_coords"]].join(
        _display_positions(lots[lots["valid_coords"]]), rsuffix="_display")

    # ------------------------------------------------------------- widgets
    lot_w = pn.widgets.Select(name="Parking lot", options=lot_ids, value="BHMBCCTHL01")
    day_w = pn.widgets.Select(name="Day", options=days, value=days[0])
    # An index-based Player plus our own label: reading times differ per day, and
    # DiscretePlayer's label keeps showing the previous day's options.
    times: list[str] = []
    player = pn.widgets.Player(
        start=0, end=1, value=0, interval=700, loop_policy="once",
        show_loop_controls=False, show_value=False,
    )
    time_label = pn.pane.Markdown("", margin=(0, 10))

    # ------------------------------------------------------- data sources
    day_src = ColumnDataSource(data=dict(t=[], occupancy_rate=[], **{k: [] for k in SERIES}))
    map_src = ColumnDataSource(data=dict(
        lon=[], lat=[], lot=[], price=[], occ=[], action=[], size=[], line=[], width=[],
        label=[],
    ))
    route_src = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[]))
    season_src = ColumnDataSource(data=dict(day=[], **{k: [] for k in SERIES}))

    # -------------------------------------------------- price + occupancy
    price_fig = figure(height=300, x_axis_type="datetime", title="Price through the day",
                       tools="xpan,xwheel_zoom,reset", y_axis_label="Price ($)")
    renderers = []
    for col, (label, color) in SERIES.items():
        renderers.append(price_fig.line("t", col, source=day_src, line_width=2,
                                        color=color, legend_label=label))
        price_fig.scatter("t", col, source=day_src, size=8, color=color,
                          line_color="white", line_width=2, legend_label=label)
    price_fig.add_tools(HoverTool(
        renderers=[renderers[0]], mode="vline",
        tooltips=[("time", "@t{%H:%M}")] + [(SERIES[c][0], f"@{c}{{$0.00}}") for c in SERIES],
        formatters={"@t": "datetime"},
    ))
    _legend_above(price_fig)
    price_fig.yaxis.formatter = NumeralTickFormatter(format="$0")
    cursor_price = Span(dimension="height", line_color=INK_MUTED, line_dash="dashed", line_width=1)
    price_fig.add_layout(cursor_price)
    _style(price_fig)

    occ_fig = figure(height=200, x_axis_type="datetime", x_range=price_fig.x_range,
                     y_range=(0, 1.05), title="Occupancy rate", tools="",
                     y_axis_label="Occupied")
    occ_fig.varea("t", y1=0, y2="occupancy_rate", source=day_src,
                  color=OCCUPANCY_COLOR, alpha=0.15)
    occ_line = occ_fig.line("t", "occupancy_rate", source=day_src, line_width=2,
                            color=OCCUPANCY_COLOR)
    occ_fig.add_layout(Span(location=cfg.full_threshold, dimension="width",
                            line_color=INK_MUTED, line_dash="dotted", line_width=1))
    occ_fig.add_tools(HoverTool(renderers=[occ_line], mode="vline",
                                tooltips=[("time", "@t{%H:%M}"),
                                          ("occupancy", "@occupancy_rate{0%}")],
                                formatters={"@t": "datetime"}))
    occ_fig.yaxis.formatter = NumeralTickFormatter(format="0%")
    cursor_occ = Span(dimension="height", line_color=INK_MUTED, line_dash="dashed", line_width=1)
    occ_fig.add_layout(cursor_occ)
    _style(occ_fig)

    # ------------------------------------------------------------------ map
    mapper = LinearColorMapper(palette=PRICE_RAMP, low=cfg.base_price * 0.6,
                               high=cfg.base_price * 1.6)
    map_fig = figure(height=470, title="Model 3 price by lot (co-located lots fanned out)",
                     tools="pan,wheel_zoom,reset", match_aspect=True)
    map_fig.add_layout(Arrow(
        source=route_src, x_start="x0", y_start="y0", x_end="x1", y_end="y1",
        line_color=REROUTE_COLOR, line_width=2, line_dash="dashed",
        end=VeeHead(size=10, line_color=REROUTE_COLOR, fill_color=REROUTE_COLOR),
    ))
    dots = map_fig.scatter("lon", "lat", source=map_src, size="size",
                           fill_color={"field": "price", "transform": mapper},
                           line_color="line", line_width="width", fill_alpha=0.9)
    map_fig.text("lon", "lat", text="label", source=map_src, x_offset=13, y_offset=0,
                 text_font_size="8pt", text_color=INK_SECONDARY, text_align="left",
                 text_baseline="middle")
    map_fig.add_tools(HoverTool(renderers=[dots], tooltips=[
        ("lot", "@lot"), ("price", "@price{$0.00}"),
        ("occupancy", "@occ{0%}"), ("action", "@action"),
    ]))
    map_fig.add_layout(ColorBar(color_mapper=mapper, width=10, title="price ($)",
                                major_label_text_color=INK_SECONDARY,
                                background_fill_color=None), "right")
    _style(map_fig)
    map_fig.ygrid.grid_line_color = None
    map_fig.axis.visible = False
    map_fig.x_range.range_padding = 0.55
    map_fig.y_range.range_padding = 0.12

    # ---------------------------------------------------- season overview
    season_fig = figure(height=240, x_axis_type="datetime",
                        title="Daily average price over the season",
                        tools="xpan,xwheel_zoom,reset", y_axis_label="Price ($)")
    season_lines = []
    for col, (label, color) in SERIES.items():
        season_lines.append(season_fig.line("day", col, source=season_src, line_width=2,
                                            color=color, legend_label=label))
    season_fig.add_tools(HoverTool(
        renderers=[season_lines[0]], mode="vline",
        tooltips=[("day", "@day{%a %d %b}")] + [(SERIES[c][0], f"@{c}{{$0.00}}") for c in SERIES],
        formatters={"@day": "datetime"},
    ))
    _legend_above(season_fig)
    season_fig.yaxis.formatter = NumeralTickFormatter(format="$0")
    _style(season_fig)

    # --------------------------------------------------------- KPI + table
    kpis = pn.pane.HTML(sizing_mode="stretch_width")
    table_title = pn.pane.Markdown("")
    table = pn.widgets.Tabulator(
        pd.DataFrame(), disabled=True, show_index=False, height=440,
        layout="fit_data_stretch",
        formatters={
            "occupancy %": {"type": "progress", "max": 100, "color": OCCUPANCY_COLOR,
                            "legend": True, "legendAlign": "left"},
        },
    )

    # ------------------------------------------------------------- updates
    def lot_day() -> pd.DataFrame:
        sel = priced[(priced["lot_id"] == lot_w.value) & (priced["day"] == day_w.value)]
        return sel.sort_values("timestamp")

    def refresh_day(*_):
        sel = lot_day()
        day_src.data = dict(
            t=sel["timestamp"], occupancy_rate=sel["occupancy_rate"],
            **{c: sel[c] for c in SERIES},
        )
        times[:] = sorted(priced.loc[priced["day"] == day_w.value, "time"].unique())
        # open on the selected lot's busiest moment of the day
        busiest = sel.loc[sel["occupancy_rate"].idxmax(), "time"] if len(sel) else times[-1]
        player.param.update(end=max(len(times) - 1, 1), value=times.index(busiest))
        price_fig.title.text = f"Price through the day - {lot_w.value}, {day_w.value}"
        refresh_instant()

    def refresh_season(*_):
        sel = priced[priced["lot_id"] == lot_w.value]
        daily = sel.groupby("day")[list(SERIES)].mean().reset_index()
        season_src.data = dict(day=pd.to_datetime(daily["day"]),
                               **{c: daily[c] for c in SERIES})
        season_fig.title.text = f"Daily average price over the season - {lot_w.value}"

    def refresh_instant(*_):
        if not times:
            return
        now_time = times[min(player.value, len(times) - 1)]
        time_label.object = f"**Time of day: {now_time}**"
        ts = pd.Timestamp(f"{day_w.value} {now_time}")
        cursor_price.location = cursor_occ.location = ts
        # each lot's latest quote at or before the cursor (same day)
        today = priced[(priced["day"] == day_w.value) & (priced["timestamp"] <= ts)]
        now = today.sort_values("timestamp").groupby("lot_id").tail(1).set_index("lot_id")

        on_map = valid_lots.join(now, how="inner", rsuffix="_q")
        sel = on_map.index == lot_w.value
        map_src.data = dict(
            lon=on_map["longitude_display"], lat=on_map["latitude_display"], lot=on_map.index,
            label=[lot.replace("Others-", "") for lot in on_map.index],
            price=on_map["price_competitive"], occ=on_map["occupancy_rate"],
            action=on_map["action"],
            size=10 + 22 * (on_map["capacity"] / lots["capacity"].max()) ** 0.5,
            line=["#0b0b0b" if s else "white" for s in sel],
            width=[3 if s else 2 for s in sel],
        )
        routes = dict(x0=[], y0=[], x1=[], y1=[])
        for _, row in on_map[on_map["reroute_to"].notna()].iterrows():
            if row["reroute_to"] in on_map.index:
                tgt = on_map.loc[row["reroute_to"]]
                x0, y0 = row["longitude_display"], row["latitude_display"]
                x1, y1 = tgt["longitude_display"], tgt["latitude_display"]
                # stop short so the arrow head is not hidden under the target dot
                routes["x0"].append(x0)
                routes["y0"].append(y0)
                routes["x1"].append(x0 + 0.88 * (x1 - x0))
                routes["y1"].append(y0 + 0.88 * (y1 - y0))
        route_src.data = routes

        tbl = now.reset_index()[[
            "lot_id", "occupancy_rate", "queue_length", "price_baseline",
            "price_demand", "price_competitive", "action", "reroute_to",
        ]].rename(columns={
            "lot_id": "lot", "occupancy_rate": "occupancy %", "queue_length": "queue",
            "price_baseline": "M1 $", "price_demand": "M2 $", "price_competitive": "M3 $",
            "reroute_to": "reroute to",
        })
        for c in ("M1 $", "M2 $", "M3 $"):
            tbl[c] = tbl[c].round(2)
        tbl["occupancy %"] = (100 * tbl["occupancy %"]).round().astype(int)
        tbl["reroute to"] = tbl["reroute to"].fillna("")
        table.value = tbl.sort_values("occupancy %", ascending=False)
        table_title.object = f"#### All lots at {now_time} on {day_w.value}"

        if lot_w.value in now.index:
            q = now.loc[lot_w.value]
            kpis.object = _kpi_html(q, lot_w.value, neighbors.get(lot_w.value, []), cfg)
        else:
            kpis.object = f"<p>No reading yet for {lot_w.value} at {now_time}.</p>"

    lot_w.param.watch(lambda e: (refresh_day(), refresh_season()), "value")
    day_w.param.watch(refresh_day, "value")
    player.param.watch(refresh_instant, "value")
    refresh_day()
    refresh_season()

    about = pn.pane.Markdown(
        f"""
**Model 1** drifts from ${cfg.base_price:.0f} by
{cfg.baseline_alpha} x (occupancy - {cfg.target_occupancy:.0%}) per reading.
**Model 2** prices a weighted demand score (occupancy, queue, traffic,
special day, vehicle type, trend), EMA-smoothed.
**Model 3** nudges Model 2 toward neighbours within {cfg.neighbor_radius_km} km;
full lots (>= {cfg.full_threshold:.0%}) hold price and reroute drivers
(red arrows on the map). Prices stay within
${cfg.min_price:.0f}-${cfg.max_price:.0f}.
""",
        styles={"font-size": "0.9em"},
    )

    template = pn.template.FastListTemplate(
        title="Urban Parking - Dynamic Pricing",
        sidebar=[lot_w, day_w, time_label, player, about],
        main=[
            kpis,
            pn.Row(pn.Column(pn.pane.Bokeh(price_fig), pn.pane.Bokeh(occ_fig)),
                   pn.pane.Bokeh(map_fig)),
            pn.Column(table_title, table),
            pn.pane.Bokeh(season_fig),
        ],
        accent="#2a78d6",
        theme_toggle=False,
        sidebar_width=300,
    )
    return template


def _kpi_html(q: pd.Series, lot: str, nbrs: list, cfg: PricingConfig) -> str:
    def tile(label: str, value: str, note: str = "") -> str:
        return (
            "<div style='flex:1;min-width:150px;padding:12px 16px;border:1px solid #e6e5e1;"
            "border-radius:8px'>"
            f"<div style='font-size:12px;color:{INK_SECONDARY}'>{label}</div>"
            f"<div style='font-size:26px;font-weight:600;color:#0b0b0b'>{value}</div>"
            f"<div style='font-size:12px;color:{INK_MUTED}'>{note}</div></div>"
        )

    action = q["action"].replace("_", " ")
    note = f"send drivers to {q['reroute_to']}" if isinstance(q["reroute_to"], str) else (
        f"{len(nbrs)} competitor(s) within {cfg.neighbor_radius_km} km"
    )
    comp = q["competitor_price"]
    comp_note = "no competitors" if pd.isna(comp) else f"neighbours avg ${comp:.2f}"
    return (
        "<div style='display:flex;gap:12px;flex-wrap:wrap'>"
        + tile(f"Model 3 price - {lot}", f"${q['price_competitive']:.2f}", comp_note)
        + tile("Model 2 price", f"${q['price_demand']:.2f}", f"demand {q['demand']:.2f}")
        + tile("Occupancy", f"{q['occupancy_rate']:.0%}",
               f"{int(q['occupancy'])} / {int(q['capacity'])} spaces, queue {int(q['queue_length'])}")
        + tile("Decision", action, note)
        + "</div>"
    )


def serve(
    port: int = 5006,
    address: str = "localhost",
    show: bool = True,
    cfg: PricingConfig = DEFAULT_CONFIG,
) -> None:
    _load(cfg)  # price the dataset once, before the first browser session
    # when bound to all interfaces (e.g. inside Docker) accept the forwarded host
    origin = "*" if address in ("0.0.0.0", "::") else None
    pn.serve(lambda: create_app(cfg), port=port, address=address, show=show,
             title="Parking Dynamic Pricing", websocket_origin=origin)


if __name__.startswith("bokeh"):  # `panel serve parking_pricing/dashboard.py`
    create_app().servable()
