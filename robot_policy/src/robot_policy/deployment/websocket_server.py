from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Any

from robot_policy.deployment import msgpack_numpy


class WebsocketPolicyServer:
    """Reference-compatible msgpack/WebSocket policy transport."""

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.idle_timeout = int(idle_timeout)
        self.metadata = metadata if metadata is not None else policy.metadata
        self.last_active = time.time()

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        except KeyboardInterrupt:
            logging.info("policy server stopped by operator")

    async def run(self) -> None:
        try:
            import websockets.asyncio.server
        except ImportError as exc:
            raise RuntimeError(
                "WebSocket serving requires `pip install websockets>=14`"
            ) from exc
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ) as server:
            if self.idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()

    async def _idle_watchdog(self, server) -> None:
        while True:
            await asyncio.sleep(min(5, self.idle_timeout))
            if time.time() - self.last_active > self.idle_timeout:
                server.close()
                await server.wait_closed()
                return

    async def _handler(self, websocket) -> None:
        packer = msgpack_numpy.Packer()
        logging.info("connection opened from %s", websocket.remote_address)
        await websocket.send(packer.pack(self.metadata))
        try:
            async for wire_message in websocket:
                self.last_active = time.time()
                try:
                    request = msgpack_numpy.unpackb(wire_message)
                    response = self.route_message(request)
                    await websocket.send(packer.pack(response))
                except Exception:
                    logging.exception("unhandled WebSocket request failure")
                    await websocket.send(traceback.format_exc())
        finally:
            logging.info("connection closed from %s", websocket.remote_address)

    def route_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(message, dict):
            return self._error("default", "request must be a dict", "unknown")
        request_id = message.get("request_id", "default")
        message_type = message.get("type", "infer")
        payload = message.get("payload", message)
        if message_type == "ping":
            return {
                "status": "ok",
                "ok": True,
                "type": "ping",
                "request_id": request_id,
            }
        if message_type in {"init", "metadata"}:
            return {
                "status": "ok",
                "ok": True,
                "type": "metadata",
                "request_id": request_id,
                "data": self.metadata,
            }
        if message_type == "reset":
            try:
                data = self.policy.reset(**(payload if isinstance(payload, dict) else {}))
                return self._success(request_id, "reset_result", data)
            except Exception as exc:
                return self._error(request_id, str(exc), "reset_result")
        if message_type in {"infer", "predict_action", "infer_realtime", "predict_action_realtime"}:
            if not isinstance(payload, dict):
                return self._error(request_id, "payload must be a dict", "inference_result")
            try:
                if message_type in {"infer_realtime", "predict_action_realtime"}:
                    data = self.policy.predict_action_realtime(**payload)
                else:
                    data = self.policy.predict_action(**payload)
                return self._success(request_id, "inference_result", data)
            except Exception as exc:
                logging.exception("policy inference failed (request_id=%s)", request_id)
                return self._error(request_id, str(exc), "inference_result")
        return self._error(
            request_id, f"unsupported message type {message_type!r}", "unknown"
        )

    @staticmethod
    def _success(request_id: Any, response_type: str, data: Any) -> dict[str, Any]:
        return {
            "status": "ok",
            "ok": True,
            "type": response_type,
            "request_id": request_id,
            "data": data,
        }

    @staticmethod
    def _error(request_id: Any, message: str, response_type: str) -> dict[str, Any]:
        return {
            "status": "error",
            "ok": False,
            "type": response_type,
            "request_id": request_id,
            "error": {"message": message},
        }
