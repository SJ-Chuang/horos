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

    p = sub.add_parser(
        "import", help="Import a COCO/YOLO/VOC/Darknet dataset into a project"
    )
    p.add_argument("source")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo", "voc", "darknet"])
    p.add_argument(
        "--no-copy",
        action="store_true",
        help="Reference images in place instead of copying them into the project",
    )
    p.add_argument(
        "--on-conflict",
        choices=["ask", "overwrite", "skip", "rename"],
        default="ask",
        help="What to do when a file name already exists with different content "
        "(default: ask — fail with the conflict list, importing nothing)",
    )
    p.add_argument(
        "--class-names",
        help="Comma-separated class names for Darknet datasets without _darknet.labels",
    )

    p = sub.add_parser("export", help="Export the project dataset")
    p.add_argument("out_dir")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo"], default="coco")

    p = sub.add_parser("convert", help="Convert a dataset between formats")
    p.add_argument("source")
    p.add_argument("out_dir")
    p.add_argument("--to", required=True, choices=["coco", "yolo"], dest="to_format")
    p.add_argument("--from", choices=["coco", "yolo", "voc", "darknet"], dest="from_format")

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

    p = sub.add_parser(
        "autolabel", help="Zero-shot pre-labels from text prompts (runs in foreground)"
    )
    p.add_argument("--project", required=True)
    p.add_argument(
        "--prompt",
        action="append",
        required=True,
        dest="prompts",
        metavar="CLASS=P1[,P2...]",
        help="Class and its prompt(s), repeatable: --prompt forklift=forklift,lift truck",
    )
    p.add_argument("--model", default="owlv2-base")
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--nms-iou", type=float, default=0.5)
    p.add_argument(
        "--output",
        choices=["bbox", "polygon"],
        default="bbox",
        help="polygon runs each box through SAM and writes the mask outline",
    )
    p.add_argument("--split", choices=["train", "valid", "test"])
    p.add_argument(
        "--include-annotated",
        action="store_true",
        help="Also pre-label images that already have confirmed annotations",
    )

    p = sub.add_parser(
        "train", help="Train a model (runs in a worker subprocess, streams events)"
    )
    p.add_argument("--project", required=True)
    p.add_argument("--model", default="rfdetr-nano")
    p.add_argument("--epochs", type=int, help="Omit to derive from dataset stats")
    p.add_argument("--batch-size", type=int, help="Omit to derive from memory probe")
    p.add_argument("--resolution", type=int)
    p.add_argument("--device", choices=["cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int)
    p.add_argument("--resume-from", help="Checkpoint path to continue training from")

    p = sub.add_parser("infer", help="Run a trained run's model on image(s)")
    p.add_argument("images", nargs="+")
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True, dest="run_id")
    p.add_argument("--threshold", type=float, default=0.5)

    p = sub.add_parser(
        "evaluate", help="COCO metrics for a run on its held-out split"
    )
    p.add_argument("--project", required=True)
    p.add_argument("--run", required=True, dest="run_id")
    p.add_argument("--split", choices=["train", "valid", "test"], default="test")

    sub.add_parser("models", help="List available models (with licenses)")
    sub.add_parser("capabilities", help="Show what this platform supports")

    p = sub.add_parser(
        "doctor", help="Check dependencies for this platform; --fix installs what's missing"
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Run the planned pip installs (torch on Jetson is never automated)",
    )

    p = sub.add_parser("ui", help="Start the Web API + WebUI server")
    p.add_argument(
        "project_path",
        nargs="?",
        default=None,
        metavar="project",
        help="Path to the horos project directory",
    )
    # kept for compatibility with older docs/scripts: horos ui --project <dir>
    p.add_argument("--project", dest="project_flag", help=argparse.SUPPRESS)
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
                on_conflict=args.on_conflict,
                class_names=(
                    [n.strip() for n in args.class_names.split(",")]
                    if args.class_names
                    else None
                ),
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
        elif args.command == "autolabel":
            from horos.api.autolabel import autolabel_events
            from horos.backends.base import dump_event

            prompts: dict[str, list[str]] = {}
            for entry in args.prompts:
                cls, _, plist = entry.partition("=")
                prompts[cls.strip()] = (
                    [p.strip() for p in plist.split(",")] if plist else [cls.strip()]
                )
            failed = False
            for event in autolabel_events(
                api.open_project(args.project),
                api.PromptSpec(prompts=prompts),
                model=args.model,
                threshold=args.threshold,
                nms_iou=args.nms_iou,
                output=args.output,
                split=args.split,
                only_unannotated=not args.include_annotated,
            ):
                sys.stdout.write(dump_event(event) + "\n")  # JSONL stream (E3-T3)
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "train":
            import time as time_mod

            from horos.api.train import TrainRunConfig

            project = api.open_project(args.project)
            record = api.start_training(
                project,
                TrainRunConfig(
                    model=args.model,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    resolution=args.resolution,
                    device=args.device,
                    seed=args.seed,
                    resume_from=args.resume_from,
                ),
            )
            print(f"run {record.run_id} started (pid {record.pid})", file=sys.stderr)  # noqa: T201
            seen = 0
            try:
                while True:
                    status = api.training_status(project, record.run_id, after=seen)
                    for event in status.events:
                        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
                        sys.stdout.flush()
                    seen = status.num_events
                    if status.run.state not in ("pending", "running"):
                        _emit(status.run.model_dump())
                        # the conclusion, checked even when numbers look perfect
                        _emit(api.run_verdict(project, record.run_id).model_dump())
                        return 0 if status.run.state == "completed" else 2
                    time_mod.sleep(1.0)
            except KeyboardInterrupt:
                api.stop_training(project, record.run_id)
                print(f"stopping run {record.run_id} ...", file=sys.stderr)  # noqa: T201
                return 130
        elif args.command == "infer":
            project = api.open_project(args.project)
            for image in args.images:
                prediction = api.infer_image(
                    project, args.run_id, image, threshold=args.threshold
                )
                sys.stdout.write(prediction.model_dump_json() + "\n")
                sys.stdout.flush()
        elif args.command == "evaluate":
            from horos.api.evaluate import evaluation_events
            from horos.backends.base import dump_event

            failed = False
            for event in evaluation_events(
                api.open_project(args.project), args.run_id, split=args.split
            ):
                sys.stdout.write(dump_event(event) + "\n")  # JSONL stream (R4)
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
        elif args.command == "models":
            _emit([m.model_dump() for m in api.list_models()])
        elif args.command == "capabilities":
            _emit(api.platform_capabilities().model_dump())
        elif args.command == "doctor":
            import subprocess

            report = api.doctor_report()
            plat = report.platform
            print(f"platform : {plat.os_family}/{plat.arch}"  # noqa: T201
                  f"{' (Jetson)' if plat.is_jetson else ''}  python {plat.python_version}")
            for dep in report.dependencies:
                mark = "ok " if dep.ok else "MISSING"
                extra = f"  ({dep.note})" if dep.note else ""
                print(f"  [{mark}] {dep.name:<12} {dep.installed or '-':<10} "  # noqa: T201
                      f"requires {dep.required}{extra}")
            if report.torch_cuda_available is not None:
                print(f"device   : cuda={report.torch_cuda_available} "  # noqa: T201
                      f"mps={report.torch_mps_available}")
            for action in report.manual_actions:
                print(f"manual   : {action}")  # noqa: T201
            if report.ok:
                print("Environment OK.")  # noqa: T201
                return 0
            if not args.fix:
                for command in report.fix_commands:
                    print(f"fix      : pip install {' '.join(command)}")  # noqa: T201
                print("Run 'horos doctor --fix' to install the above.")  # noqa: T201
                return 1
            for command in report.fix_commands:
                print(f"==> pip install {' '.join(command)}")  # noqa: T201
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *command], check=True
                )
            if report.manual_actions:
                print("Manual steps remain (see above) — not automated on purpose.")  # noqa: T201
                return 1
            print("Fixes applied. Re-run 'horos doctor' to verify.")  # noqa: T201
        elif args.command == "ui":
            from horos.web.app import create_app

            project_path = args.project_path or args.project_flag
            if not project_path:
                print("usage: horos ui <project>", file=sys.stderr)  # noqa: T201
                return 2
            app = create_app(project_path)
            app.run(host=args.host, port=args.port)
    except HorosError as exc:
        print(f"error [{exc.code}]: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
