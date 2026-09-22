from __future__ import annotations

import argparse
import logging
import socket

from robot_policy.deployment.policy_wrapper import PolicyServerWrapper
from robot_policy.deployment.server_config import (
    load_server_config,
    validate_server_config,
)
from robot_policy.deployment.websocket_server import WebsocketPolicyServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve any robot_policy checkpoint over the Piper WebSocket API."
    )
    parser.add_argument("--ckpt_path", "--checkpoint", dest="checkpoint", required=True)
    parser.add_argument(
        "--config-json",
        help="corresponding policy-server JSON; validates model/input contract",
    )
    parser.add_argument("--prepared-path")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--idle-timeout", type=int, default=-1)
    parser.add_argument("--task")
    parser.add_argument(
        "--binary-gripper", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--gripper-threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_server_config(args.config_json) if args.config_json else None
    prepared_path = args.prepared_path or (config.prepared_path if config else None)
    precision = args.precision or (config.precision if config else "bf16")
    task = args.task or (config.task_instruction if config else "Stack the cups.")
    binary_gripper = (
        args.binary_gripper
        if args.binary_gripper is not None
        else config.binary_gripper
        if config
        else True
    )
    gripper_threshold = (
        args.gripper_threshold
        if args.gripper_threshold is not None
        else config.gripper_threshold
        if config
        else 0.3
    )
    wrapper = PolicyServerWrapper(
        args.checkpoint,
        prepared_path=prepared_path,
        device=args.device,
        precision=precision,
        task_instruction=task,
        binary_gripper=binary_gripper,
        gripper_threshold=gripper_threshold,
    )
    if config:
        validate_server_config(config, wrapper.metadata)
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
