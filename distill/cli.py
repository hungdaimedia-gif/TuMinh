from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from distill.importers import (
    DistillImportError,
    ExternalImporter,
    RuntimeExporter,
    write_trajectory_exports,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "export-runtime":
            output = RuntimeExporter(arguments.journal_root).export(
                arguments.task_id,
                arguments.output,
            )
            print(f"Exported runtime trajectory to {output}")
            return 0
        if arguments.command == "import-external":
            trajectories = ExternalImporter().import_file(
                arguments.path,
                source_name=arguments.source_name,
                license_or_authorization=arguments.license_or_authorization,
                authorized_by=arguments.authorized_by,
            )
            if Path(arguments.path).resolve(strict=True) == arguments.output.resolve(strict=False):
                raise DistillImportError("Import output cannot overwrite the supplied source file")
            output = write_trajectory_exports(trajectories, arguments.output)
            print(f"Imported {len(trajectories)} authorized trajectory record(s) to {output}")
            return 0
        parser.error("A distillation command is required")
    except DistillImportError as error:
        print(f"Distillation import error: {error}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explicit, local trajectory export and authorized import."
    )
    commands = parser.add_subparsers(dest="command")

    export_runtime = commands.add_parser("export-runtime")
    export_runtime.add_argument(
        "--journal-root",
        type=Path,
        default=Path("memory") / "trajectories",
    )
    export_runtime.add_argument("--task-id", required=True)
    export_runtime.add_argument("--output", type=Path, required=True)

    import_external = commands.add_parser("import-external")
    import_external.add_argument("path", type=Path)
    import_external.add_argument("--source-name", required=True)
    import_external.add_argument("--license-or-authorization", required=True)
    import_external.add_argument("--authorized-by", required=True)
    import_external.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
