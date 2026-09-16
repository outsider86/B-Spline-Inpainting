from __future__ import annotations

import os
from typing import Any
import uuid

import numpy as np

from robot_policy.deployment import msgpack_numpy


class PolicyClient:
    """Small synchronous client for the deployment server.

    Standard ``infer`` responses remain compatible with the prior Piper
    client. This client additionally retains ``normalized_control_rows``,
    which is required to use B-spline ttRTC safely.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 10093, timeout: float = 30.0):
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError("PolicyClient requires `pip install websockets>=14`") from exc
        for key in (
            "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"
        ):
            os.environ.pop(key, None)
        self._connection = connect(
            f"ws://{host}:{int(port)}",
            compression=None,
            max_size=None,
            open_timeout=timeout,
            ping_interval=None,
        )
        hello = self._connection.recv(timeout=timeout)
        if isinstance(hello, str):
            raise RuntimeError(f"server returned text during handshake: {hello}")
        self.metadata = msgpack_numpy.unpackb(hello)
        self.timeout = timeout

    def close(self) -> None:
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def request(self, message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        self._connection.send(
            msgpack_numpy.packb(
                {"type": message_type, "request_id": request_id, "payload": payload}
            )
        )
        wire = self._connection.recv(timeout=self.timeout)
        if isinstance(wire, str):
            raise RuntimeError(f"policy server returned text error:\n{wire}")
        response = msgpack_numpy.unpackb(wire)
        if response.get("request_id") != request_id:
            raise RuntimeError("policy server response request_id mismatch")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", {}).get("message", "inference failed"))
        return response["data"]

    def predict_action(
        self,
        examples: list[dict[str, Any]],
        *,
        state_coordinates: str = "normalized",
        seed: int = 20260915,
    ) -> dict[str, Any]:
        return self.request(
            "infer",
            {
                "examples": examples,
                "unnorm_key": "new_embodiment",
                "state_coordinates": state_coordinates,
                "seed": int(seed),
            },
        )

    def predict_action_realtime(
        self,
        examples: list[dict[str, Any]],
        *,
        inference_delay: int,
        previous: np.ndarray,
        state_coordinates: str = "normalized",
        seed: int = 20260915,
    ) -> dict[str, Any]:
        field = self.metadata.get("rtc_requires_previous_field")
        if field not in {"prev_action_chunk", "prev_control_rows"}:
            raise RuntimeError("server does not advertise a valid RTC conditioning field")
        payload = {
            "examples": examples,
            "inference_delay": int(inference_delay),
            "unnorm_key": "new_embodiment",
            "state_coordinates": state_coordinates,
            "seed": int(seed),
            field: np.asarray(previous, dtype=np.float32),
        }
        return self.request("infer_realtime", payload)
