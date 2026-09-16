from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from typing import Any

import torch


@dataclass(frozen=True)
class TimedPlan:
    actions: torch.Tensor
    observation_timestamp: float
    inference_started: float
    inference_finished: float
    episode_id: int

    @property
    def observation_age_ms(self) -> float:
        return 1000 * (self.inference_finished - self.observation_timestamp)

    @property
    def inference_delay_ms(self) -> float:
        return 1000 * (self.inference_finished - self.inference_started)


class ActionExecutor:
    """Timestamped dataset/mock executor; this class has no hardware I/O."""
    def __init__(self, frequency_hz: float = 30.0, timeout_s: float = 1.0, stale_plan_s: float = 0.5):
        self.frequency_hz=frequency_hz; self.timeout_s=timeout_s; self.stale_plan_s=stale_plan_s
        self.queue: deque[tuple[float,torch.Tensor]] = deque(); self.episode_id: int | None=None; self.last_command_time: float | None=None
        self.timeline: list[dict[str,Any]]=[]

    def reset(self, episode_id: int | None=None) -> None:
        self.queue.clear(); self.episode_id=episode_id; self.last_command_time=None; self.timeline.clear()

    def submit(self, plan: TimedPlan, now: float, committed_steps: int) -> bool:
        if self.episode_id != plan.episode_id: self.reset(plan.episode_id)
        if now-plan.observation_timestamp > self.stale_plan_s:
            self.timeline.append({"event":"stale_plan_rejected","time":now}); return False
        keep=min(committed_steps,len(self.queue)); committed=[self.queue.popleft() for _ in range(keep)]
        self.queue.clear(); dt=1/self.frequency_hz
        for item in committed: self.queue.append(item)
        start=now+keep*dt
        for i,action in enumerate(plan.actions[committed_steps:]): self.queue.append((start+i*dt,action.detach().cpu()))
        self.timeline.append({"event":"plan_switch","time":now,"committed_steps":keep,"observation_age_ms":plan.observation_age_ms,"inference_delay_ms":plan.inference_delay_ms})
        return True

    def command(self, now: float) -> torch.Tensor | None:
        if self.last_command_time is not None and now-self.last_command_time > self.timeout_s:
            self.queue.clear(); self.timeline.append({"event":"timeout","time":now})
        if not self.queue:
            self.timeline.append({"event":"empty_queue","time":now}); return None
        _,action=self.queue.popleft(); self.last_command_time=now; self.timeline.append({"event":"command","time":now})
        return action

    def interrupt(self, now: float) -> None:
        self.queue.clear(); self.timeline.append({"event":"interrupt","time":now})
