"""horos CLI (E9-T5): the full workflow without a browser (E9-S3).

This is an interface layer like horos.web — it may print (it IS the output
device) but all logic lives in horos.api (R2).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import horos
import horos.api as api
from horos.errors import HorosError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="horos",
        description="horos: annotate, train, evaluate, deploy perception models.",
    )
    parser.add_argument("--version", action="version", version=horos.__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create a new horos project")
    p.add_argument("path")
    p.add_argument("--name")

    p = sub.add_parser("import", help="Import a COCO/YOLO dataset into a project")
    p.add_argument("source")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo"])
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Reference images in place instead of copying them into the project",
    )

    p = sub.add_parser("export", help="Export the project dataset")
    p.add_argument("out_dir")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo"], default="coco")

    p = sub.add_parser("convert", help="Convert a dataset between formats")
    p.add_argument("source")
    p.add_argument("out_dir")
    p.add_argument("--to", required=True, choices=["coco", "yolo"], dest="to_format")
    p.add_argument("--from", choices=["coco", "yolo"], dest="from_format")

    p = sub.add_parser("validate", help="Validate the project dataset")
    p.add_argument("--project", required=True)

    p = sub.add_parser("stats", help="Show dataset statistics")
    p.add_argument("--project", required=True)

    p = sub.add_parser("split", help="Re-split images into train/valid/test")
    p.add_argument("--project", required=True)
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--valid", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    sub.add_parser("models", help="List available models (with licenses)")
    sub.add_parser("capabilities", help="Show what this platform supports")

    p = sub.add_parser("ui", help="Start the Web API + WebUI server")
    p.add_argument("--project", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)

    return parser


def _emit(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))  # noqa: T201


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "init":
            project = api.create_project(args.path, name=args.name)
            _emit({"root": str(project.root), "name": project.manifest.name})
        elif args.command == "import":
            summary = api.import_dataset(
                api.open_project(args.project),
                args.source,
                format=args.format,
                copy_images=not args.no_copy,
            )
            _emit(summary.model_dump())
        elif args.command == "export":
            written = api.export_dataset(
                api.open_project(args.project), args.out_dir, format=args.format
            )
            _emit({"path": str(written)})
        elif args.command == "convert":
            written = api.convert_dataset(
                args.source, args.out_dir,
                to_format=args.to_format, from_format=args.from_format,
            )
            _emit({"path": str(written)})
        elif args.command == "validate":
            report = api.validate_project(api.open_project(args.project))
            _emit(report.model_dump() | {"ok": report.ok})
            return 0 if report.ok else 1
        elif args.command == "stats":
            _emit(api.dataset_stats(api.open_project(args.project)).model_dump())
        elif args.command == "split":
            counts = api.resplit(
                api.open_project(args.project),
                train=args.train, valid=args.valid, test=args.test, seed=args.seed,
            )
            _emit(counts)
        elif args.command == "models":
            _emit([m.model_dump() for m in api.list_models()])
        elif args.command == "capabilities":
            _emit(api.platform_capabilities().model_dump())
        elif args.command == "ui":
            from horos.web.app import create_app

            app = create_app(args.project)
            app.run(host=args.host, port=args.port)
    except HorosError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
