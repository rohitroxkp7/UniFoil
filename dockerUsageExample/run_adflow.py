#!/usr/bin/env python3
import argparse
import sys

import docker


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run mpirun inside an existing Docker container."
    )
    parser.add_argument("--container", default="adock", help="Container name")
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start the container if it is not running",
    )
    parser.add_argument("--np", type=int, default=16, help="MPI processes")
    parser.add_argument(
        "--workdir",
        default="/home/mdolabuser/mount",
        help="Workdir inside container",
    )
    parser.add_argument(
        "--script",
        default="aero_run.py",
        help="Python script to run inside the container",
    )
    parser.add_argument(
        "--env",
        default="/home/mdolabuser/.bashrc_mdolab",
        help="Shell file to source before running (set to '' to skip)",
    )
    args = parser.parse_args()

    client = docker.from_env()
    try:
        container = client.containers.get(args.container)
    except docker.errors.NotFound:
        print(f"Container not found: {args.container}", file=sys.stderr)
        return 2

    if container.status != "running":
        if args.start:
            container.start()
            container.reload()
        else:
            print(
                f"Container is not running: {args.container}. "
                f"Start it first or pass --start.",
                file=sys.stderr,
            )
            return 3

    pre = f"source {args.env} && " if args.env else ""
    cmd_str = f"{pre}cd {args.workdir} && mpirun -np {args.np} python {args.script}"
    cmd = "bash -lc " + repr(cmd_str)

    # Stream output live
    api = container.client.api
    exec_id = api.exec_create(container.id, cmd)["Id"]
    for stdout, stderr in api.exec_start(exec_id, stream=True, demux=True):
        if stdout:
            sys.stdout.write(stdout.decode(errors="replace"))
            sys.stdout.flush()
        if stderr:
            sys.stderr.write(stderr.decode(errors="replace"))
            sys.stderr.flush()

    exit_code = api.exec_inspect(exec_id).get("ExitCode", 1)
    return int(exit_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
