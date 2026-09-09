"""horos CLI (E9-T5): the full workflow without a browser (E9-S3).

This is an interface layer like horos.web — it may print (it IS the output
device) but all logic lives in horos.api (R2).
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path

import horos
import horos.api as api
from horos.errors import HorosError, ProjectError

MANIFEST_NAME = "horos.json"


def find_project_root(start: Path | str | None = None) -> Path | None:
    """The nearest horos project at or above `start` (default: the cwd).

    Lets every project command be run from inside the project — the same way
    git works — instead of repeating --project on each invocation."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    return None


def _project_arg(args, attribute: str = "project"):
    """Open the project named by the flag, else the one containing the cwd."""
    explicit = getattr(args, attribute, None)
    if explicit:
        return api.open_project(explicit)
    root = find_project_root()
    if root is None:
        raise ProjectError(
            f"No horos project here: run 'horos {args.command}' from inside a "
            f"project directory (one containing {MANIFEST_NAME}), or pass "
            f"--project <dir>. 'horos init' creates one."
        )
    return api.open_project(root)


def _resolve_run(project, run_id: str | None, *, need_checkpoint: bool = True) -> str:
    """`--run` defaults to the newest usable run of the project.

    With `need_checkpoint` (infer, evaluate, export-model) that means the
    newest completed run that actually has weights; report accepts any run."""
    if run_id:
        return run_id
    runs = api.list_runs(project)
    if not runs:
        raise ProjectError(
            f"Project '{project.manifest.name}' has no training runs yet — "
            f"run 'horos train' first."
        )
    usable = [
        r for r in runs
        if not need_checkpoint or (r.state == "completed" and r.checkpoint)
    ]
    if not usable:
        states = ", ".join(sorted({r.state for r in runs}))
        raise ProjectError(
            f"No completed training run with a checkpoint in project "
            f"'{project.manifest.name}' (runs are: {states}). Pass --run <id> "
            f"to pick one explicitly."
        )
    chosen = usable[0]  # list_runs is newest first
    print(f"using run {chosen.run_id} ({chosen.model})", file=sys.stderr)  # noqa: T201
    return chosen.run_id



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
        "import", help="Import a COCO/YOLO/VOC/Darknet/VIA dataset into a project"
    )
    p.add_argument("source")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo", "voc", "darknet", "via", "labelme"])
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
        help="Comma-separated class names for Darknet datasets without "
        "_darknet.labels, or VIA datasets without class attributes",
    )

    p = sub.add_parser("export", help="Export the project dataset")
    p.add_argument("out_dir")
    p.add_argument("--project", required=True)
    p.add_argument("--format", choices=["coco", "yolo", "labelme"], default="coco")

    p = sub.add_parser("convert", help="Convert a dataset between formats")
    p.add_argument("source")
    p.add_argument("out_dir")
    p.add_argument("--to", required=True, choices=["coco", "yolo", "labelme"], dest="to_format")
    p.add_argument("--from", choices=["coco", "yolo", "voc", "darknet", "via", "labelme"],
                   dest="from_format")

    p = sub.add_parser("validate", help="Validate the project dataset")
    p.add_argument("--project", required=True)
    p.add_argument(
        "--fix",
        action="store_true",
        help="Clamp auto-fixable out-of-bounds boxes (small annotation-tool "
        "overshoots) back into their images, then re-validate",
    )

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
    p.add_argument(
        "--classes",
        help="Comma-separated category names to train on (default: all); "
        "objects of unselected classes become background",
    )
    p.add_argument(
        "--include-background",
        action="store_true",
        help="With --classes: keep images that contain none of the selected "
        "classes as background negatives (default: drop them)",
    )

    p = sub.add_parser(
        "report", help="Render a run's training report (16:9 PNG dashboard, PDF, or Excel)"
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--format", choices=["png", "pdf", "xlsx"], default="png")
    p.add_argument("--out", help="Output file (default: <run>/exports/training_report.<format>)")

    p = sub.add_parser(
        "export-model",
        help="Export a completed run's model with its model card (streams events)",
    )
    p.add_argument(
        "--project",
        help="Project directory (default: the project containing the current directory)",
    )
    p.add_argument(
        "--run",
        dest="run_id",
        help="Training run id (default: the newest completed run of this project)",
    )
    p.add_argument("--format", choices=["pytorch", "onnx", "tensorrt"], default="onnx")
    p.add_argument("--dynamic-batch", action="store_true", help="ONNX: dynamic batch axis")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")

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
        "install",
        help="Install the ML stack (torch, rfdetr, albumentations, transformers, "
        "plus the ONNX export and report libraries) matched to this machine",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU-only torch build even if an NVIDIA GPU is present",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned pip commands without running them",
    )
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Also install NVIDIA's TensorRT wheels for this GPU (NVIDIA license; "
        "needed for TensorRT engine export)",
    )

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


#: commands that cannot run without the ML stack `horos install` provides
_ML_GATED_COMMANDS = frozenset({"autolabel", "train", "infer", "evaluate", "export-model"})


def _ml_preflight(command: str) -> int | None:
    """Fail fast (with the fix) when an ML command lacks its dependencies.

    `pip install horos` ships without torch/rfdetr/transformers on purpose;
    this is the moment the gap becomes the user's problem, so this is where
    the answer must be. `ui` only warns — dataset management and annotation
    work without the ML stack.
    """
    from horos.api.install import check_ml_ready

    readiness = check_ml_ready()
    for message in readiness.warnings:
        print(f"warning: {message}", file=sys.stderr)  # noqa: T201
    if not readiness.missing:
        return None
    names = ", ".join(readiness.missing)
    if command == "ui":
        print(  # noqa: T201
            f"warning: ML dependencies are not installed ({names}) — "
            "autolabel, training and inference will be unavailable. "
            "Run 'horos install' to add them.",
            file=sys.stderr,
        )
        return None
    print(  # noqa: T201
        f"error [ml-not-installed]: 'horos {command}' needs the ML stack, "
        f"but these packages are missing: {names}.\n"
        "Run 'horos install' — it detects your platform and GPU and installs "
        "the matching builds ('horos install --cpu' forces CPU-only).",
        file=sys.stderr,
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in _ML_GATED_COMMANDS or args.command == "ui":
        exit_code = _ml_preflight(args.command)
        if exit_code is not None:
            return exit_code
    try:
        if args.command == "init":
            project = api.create_project(args.path, name=args.name)
            _emit({"root": str(project.root), "name": project.manifest.name})
        elif args.command == "import":
            last_phase = [""]

            def report(event) -> None:
                # one stderr line per phase change plus the final tick of each
                if event.type != "progress":
                    return
                done = event.total is not None and event.current == event.total
                if event.phase != last_phase[0] or done:
                    last_phase[0] = event.phase
                    count = f" {event.current}/{event.total}" if event.total else ""
                    note = f" ({event.message})" if event.message else ""
                    print(f"{event.phase}{count}{note}", file=sys.stderr)  # noqa: T201

            project = api.open_project(args.project)
            names = (
                [n.strip() for n in args.class_names.split(",")] if args.class_names else None
            )
            if zipfile.is_zipfile(args.source):
                summary = api.import_zip(
                    project,
                    args.source,
                    on_conflict=args.on_conflict,
                    class_names=names,
                    progress=report,
                )
            else:
                summary = api.import_dataset(
                    project,
                    args.source,
                    format=args.format,
                    copy_images=not args.no_copy,
                    on_conflict=args.on_conflict,
                    class_names=names,
                    progress=report,
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
            project = api.open_project(args.project)
            if args.fix:
                result = api.fix_validation_issues(project)
                _emit(result.model_dump() | {"ok": result.report.ok})
                return 0 if result.report.ok else 1
            report = api.validate_project(project)
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
                    categories=(
                        [c.strip() for c in args.classes.split(",")]
                        if args.classes
                        else None
                    ),
                    include_background=args.include_background,
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
        elif args.command == "report":
            project = _project_arg(args)
            # a report is readable for any run, finished or not
            run_id = _resolve_run(project, args.run_id, need_checkpoint=False)
            path = api.export_training_report(
                project, run_id, format=args.format, out_path=args.out,
            )
            _emit({"path": str(path), "format": args.format})
        elif args.command == "export-model":
            from horos.api.export import model_export_events

            project = _project_arg(args)
            failed = False
            for event in model_export_events(
                project, _resolve_run(project, args.run_id), format=args.format,
                options={"dynamic_batch": args.dynamic_batch, "opset": args.opset},
            ):
                sys.stdout.write(event.model_dump_json() + "\n")
                sys.stdout.flush()
                failed = failed or event.type == "failed"
            if failed:
                return 2
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
        elif args.command == "install":
            import subprocess

            from horos.api.install import plan_install

            plan = plan_install(cpu=args.cpu, tensorrt=args.tensorrt)
            plat = plan.platform
            print(f"platform : {plat.os_family}/{plat.arch}"  # noqa: T201
                  f"{' (Jetson)' if plat.is_jetson else ''}  python {plat.python_version}")
            print(f"cuda     : driver supports {plan.cuda_version}"  # noqa: T201
                  if plan.cuda_version else "cuda     : no NVIDIA GPU detected")
            for note in plan.notes:
                print(f"note     : {note}")  # noqa: T201
            if plan.empty:
                print("ML stack already installed — nothing to do.")  # noqa: T201
                return 0
            for command in plan.pip_commands:
                print(f"plan     : pip install {' '.join(command)}")  # noqa: T201
            for action in plan.manual_actions:
                print(f"manual   : {action}")  # noqa: T201
            if args.dry_run:
                return 0
            for command in plan.pip_commands:
                print(f"==> pip install {' '.join(command)}")  # noqa: T201
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", *command], check=True
                )
            if plan.manual_actions:
                print("Manual steps remain (see above) — not automated on purpose.")  # noqa: T201
                return 1
            print("ML stack installed. Run 'horos doctor' to verify.")  # noqa: T201
        elif args.command == "doctor":
            import subprocess

            report = api.doctor_report()
            plat = report.platform
            print(f"platform : {plat.os_family}/{plat.arch}"  # noqa: T201
                  f"{' (Jetson)' if plat.is_jetson else ''}  python {plat.python_version}")
            for dep in report.dependencies:
                # BAD = installed but wrong (e.g. a CPU torch on a GPU machine)
                mark = "ok " if dep.ok else ("BAD" if dep.installed else "MISSING")
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
                print("Run 'horos install' (or 'horos doctor --fix') to install the above.")  # noqa: T201
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
