import copy

import torch

from robot_policy.config import Config
from robot_policy.policies import create_policy
from robot_policy.policies.common import parameter_groups


def batch():
    return {
        "vision_features": torch.randn(2, 2, 16, 2176),
        "state": torch.randn(2, 7),
        "continuous_target": torch.randn(2, 18, 7),
        "discrete_target": torch.randint(0, 256, (2, 18, 7)),
        "control_valid_mask": torch.ones(2, 18, 7, dtype=torch.bool),
    }


def raw_batch():
    data=batch()
    data["continuous_target"]=torch.randn(2,30,7)
    data["discrete_target"]=torch.randint(0,256,(2,30,7))
    data["control_valid_mask"]=torch.ones(2,30,7,dtype=torch.bool)
    return data


def test_all_architectures_forward_backward_sample():
    for architecture in ("fm", "discrete_layerwise", "discrete_joint"):
        cfg = Config(); cfg.policy.architecture = architecture; cfg.policy.hidden_dim = 48; cfg.policy.heads = 4; cfg.policy.depth = 2; cfg.policy.block_size = 21
        model = create_policy(cfg); data = batch()
        output = model.loss(data); output["loss"].backward()
        assert torch.isfinite(output["loss"])
        assert torch.isfinite(output["action_mse"]) and output["action_mse"] >= 0
        assert model.observation.vision_projector[1].weight.grad.abs().sum() > 0
        predicted = model.sample(data, steps=2, rounds=2)
        assert predicted.shape == (2, 18, 7)


def test_observation_changes_logits():
    cfg = Config(); cfg.policy.architecture = "discrete_layerwise"; cfg.policy.hidden_dim = 48; cfg.policy.heads = 4; cfg.policy.depth = 2
    model = create_policy(cfg).eval(); data = batch(); tokens = torch.full((2,126),256)
    obs1 = model.observations(data); logits1 = model.logits(tokens, obs1)
    changed = copy.deepcopy(data); changed["state"] += 5
    logits2 = model.logits(tokens, model.observations(changed))
    assert not torch.allclose(logits1, logits2)


def test_joint_attention_topology_and_cache_equivalence():
    cfg = Config(); cfg.policy.architecture = "discrete_joint"; cfg.policy.hidden_dim = 48; cfg.policy.heads = 4; cfg.policy.depth = 2; cfg.policy.block_size = 21
    model = create_policy(cfg).eval(); data = batch(); obs = model.observations(data)
    mask = model.attention_mask(33, 42, torch.device("cpu"))
    assert mask[:33, :33].all() and not mask[:33, 33:].any()
    assert mask[33:54, :54].all() and not mask[33:54, 54:].any()
    tokens = torch.randint(0, 256, (2, 42)); full = model.logits(tokens, obs)
    _, cache = model.logits(tokens[:, :21], obs, capture_cache_prefix=54)
    cached = model.cached_suffix_logits(tokens[:, 21:42], obs, cache)
    torch.testing.assert_close(cached, full[:, 21:42], atol=2e-5, rtol=2e-5)
    changed = batch(); changed_obs = model.observations(changed)
    try:
        model.cached_suffix_logits(tokens[:, 21:42], changed_obs, cache)
    except ValueError:
        pass
    else:
        raise AssertionError("changed observation must invalidate cache")


def test_joint_cached_sampling_covers_remasking_and_block_transitions():
    cfg = Config(); cfg.policy.architecture = "discrete_joint"; cfg.policy.hidden_dim = 24; cfg.policy.heads = 4; cfg.policy.depth = 1; cfg.policy.block_size = 21
    model = create_policy(cfg).eval(); data = {k:v[:1] for k,v in batch().items()}
    fixed=torch.zeros(1,18,7,dtype=torch.bool); fixed[:,:4]=True
    prefix=torch.randint(0,256,(1,18,7)); torch.manual_seed(9)
    reference=model.sample(data,rounds=3,prefix_values=prefix,fixed_mask=fixed,use_cache=False)
    torch.manual_seed(9)
    cached,trace=model.sample(data,rounds=3,prefix_values=prefix,fixed_mask=fixed,use_cache=True,return_trace=True)
    torch.manual_seed(9)
    legacy=model.sample(data,rounds=3,prefix_values=prefix,fixed_mask=fixed,use_cache=True,fuse_cache_transition=False)
    assert torch.equal(reference,cached)
    assert torch.equal(reference,legacy)
    assert all(torch.equal(item["tokens"].reshape(1,18,7)[fixed],prefix[fixed]) for item in trace)
    assert {item["block_start"] for item in trace} == {0,21,42,63,84,105}


def test_default_trainable_parameter_budgets_within_ten_percent():
    totals=[]
    for architecture in ("fm","discrete_layerwise","discrete_joint"):
        cfg=Config(); cfg.policy.architecture=architecture; totals.append(parameter_groups(create_policy(cfg))["total_trainable"])
    assert (max(totals)-min(totals))/max(totals) <= .10


def test_raw_action_architectures_forward_backward_and_sample():
    for architecture in ("fm","discrete_layerwise","discrete_joint"):
        cfg=Config(); cfg.data.action_representation="raw"; cfg.policy.architecture=architecture
        cfg.policy.hidden_dim=24; cfg.policy.heads=4; cfg.policy.depth=1; cfg.policy.block_size=21
        model=create_policy(cfg); data=raw_batch(); result=model.loss(data); result["loss"].backward()
        prediction=model.sample(data,steps=2,rounds=2)
        assert prediction.shape==(2,30,7)
        assert torch.isfinite(result["loss"])
        assert torch.isfinite(result["action_mse"]) and result["action_mse"] >= 0


def test_raw_discrete_action_mse_matches_supervised_definition():
    cfg=Config(); cfg.data.action_representation="raw"; cfg.policy.architecture="discrete_layerwise"
    cfg.policy.hidden_dim=24; cfg.policy.heads=4; cfg.policy.depth=1
    model=create_policy(cfg); data=raw_batch(); logits=torch.randn(2,210,256)
    supervised=torch.rand(2,210) > .45
    bins=torch.arange(256,dtype=torch.float32)
    expected=(logits.float().softmax(-1)*bins).sum(-1)/255*2-1
    reference=((expected-data["continuous_target"].flatten(1))**2)[supervised].mean()
    torch.testing.assert_close(model.logits_action_mse(logits,data,supervised),reference)
