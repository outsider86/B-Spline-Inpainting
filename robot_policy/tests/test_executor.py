import torch

from robot_policy.inference.executor import ActionExecutor, TimedPlan


def test_executor_empty_stale_timeout_interrupt_and_episode_reset():
    ex=ActionExecutor(timeout_s=.1,stale_plan_s=.2)
    assert ex.command(0.0) is None
    stale=TimedPlan(torch.zeros(3,7),0.0,0.0,.3,1)
    assert not ex.submit(stale,.3,0)
    plan=TimedPlan(torch.arange(21).reshape(3,7),1.0,1.0,1.01,1)
    assert ex.submit(plan,1.01,0); assert ex.command(1.01) is not None
    assert ex.command(1.5) is None
    ex.submit(TimedPlan(torch.ones(2,7),2.0,2.0,2.01,1),2.01,0); ex.interrupt(2.02); assert ex.command(2.03) is None
    ex.submit(TimedPlan(torch.ones(2,7),3.0,3.0,3.01,2),3.01,0)
    assert ex.episode_id==2 and len(ex.queue)==2
