"""Command line entry point: ``parking-pricing <command>`` or ``python -m parking_pricing``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import DEFAULT_CONFIG, PricingConfig
from .data import DEFAULT_DATASET, clean, load_raw


def _config(path: str | None) -> PricingConfig:
    return PricingConfig.from_json(path) if path else DEFAULT_CONFIG


def cmd_replay(args: argparse.Namespace) -> None:
    import pandas as pd

    from . import metrics
    from .replay import run_replay

    cfg = _config(args.config)
    df, report = clean(load_raw(args.dataset))
    print(report, end="\n\n")
    priced = run_replay(df, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    priced.to_csv(out / "prices.csv", index=False)
    summary = metrics.model_summary(priced, cfg)
    summary.to_csv(out / "model_summary.csv")
    lots = metrics.lot_summary(priced)
    lots.to_csv(out / "lot_summary.csv")

    with pd.option_context("display.width", 140, "display.max_columns", 20,
                           "display.float_format", "{:,.3f}".format):
        print("Model comparison\n", summary, "\n", sep="")
        print("Per lot\n", lots, "\n", sep="")
        print("Model 3 decisions (%)\n", metrics.action_breakdown(priced).to_string(), sep="")
    print(f"\nwrote {out / 'prices.csv'}, model_summary.csv, lot_summary.csv")


def cmd_stream(args: argparse.Namespace) -> None:
    from .pathway_pipeline import run_stream

    df, report = clean(load_raw(args.dataset))
    print(report, end="\n\n")
    out = run_stream(df, args.out, _config(args.config),
                     input_rate=args.rate, mode="static" if args.static else "replay")
    print(f"wrote {out / 'live_prices_stream.csv'} and {out / 'daily_summary_stream.csv'}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    from .dashboard import serve

    serve(port=args.port, address=args.address, show=not args.no_browser,
          cfg=_config(args.config))


def cmd_api(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(_config(args.config), dataset=args.dataset),
                host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="parking-pricing", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, out=True):
        sp.add_argument("--dataset", default=str(DEFAULT_DATASET))
        sp.add_argument("--config", help="JSON file overriding PricingConfig fields")
        if out:
            sp.add_argument("--out", default="outputs")

    sp = sub.add_parser("replay", help="price the dataset with the in-process engine")
    common(sp)
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("stream", help="run the Pathway streaming pipeline (Linux/macOS)")
    common(sp)
    sp.add_argument("--rate", type=float, default=1000.0, help="replay speed, rows/second")
    sp.add_argument("--static", action="store_true", help="read all rows at once")
    sp.set_defaults(func=cmd_stream)

    sp = sub.add_parser("dashboard", help="serve the Panel dashboard")
    common(sp, out=False)
    sp.add_argument("--port", type=int, default=5006)
    sp.add_argument("--address", default="localhost")
    sp.add_argument("--no-browser", action="store_true")
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("api", help="serve the REST API")
    common(sp, out=False)
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(func=cmd_api)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        args.func(args)
    except RuntimeError as exc:  # environment problems, e.g. Pathway missing
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
