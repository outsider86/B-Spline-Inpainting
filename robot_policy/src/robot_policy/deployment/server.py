from __future__ import annotations

import argparse
import logging
import socket

from robot_policy.deployment.policy_wrapper import PolicyServerWrapper
from robot_policy.deployment.websocket_server import WebsocketPolicyServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve any robot_policy checkpoint over the Piper WebSocket API."
    )
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument("--prepared-path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--idle-timeout", type=int, default=-1)
    parser.add_argument("--task", default="Stack the cups.")
    parser.add_argument(
        "--binary-gripper", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.3)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    wrapper = PolicyServerWrapper(
        args.checkpoint,
        prepared_path=args.prepared_path,
        device=args.device,
        precision=args.precision,
        task_instruction=args.task,
        binary_gripper=args.binary_gripper,
        gripper_threshold=args.gripper_threshold,
    )
    logging.warning(
        "[POLICY SERVER] host=%s bind=%s:%d metadata=%s",
        socket.gethostname(),
        args.host,
        args.port,
        wrapper.metadata,
    )
    WebsocketPolicyServer(
        wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

