import numpy as np
import torch
from types import SimpleNamespace

from robot_policy.config import Config
from robot_policy.encoders.bspline_adapter import BSplineAdapter
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.pigdm import hard_mask_pigdm_sample
from robot_policy.rtc.training import RawActionCodec, make_reference_ttrtc_condition
from robot_policy.training import (
    _cached_parent_prediction,
    _previous_batch,
    _sample_training_delays,
)


def test_required_geometry_and_left_clamp():
    adapter = BSplineAdapter(Config())
    assert adapter.config.degree == 3
    assert adapter.config.num_basis == 18
    assert adapter.config.executable_spans == 15
    assert adapter.config.right_context_spans == 3
    assert adapter.encoder.knots.tolist()[:4] == [0.0] * 4
    values = np.random.default_rng(0).normal(size=(30, 7))
    result = adapter.encoder.encode_chunk(values)
    assert result.control_points.shape == (18, 7)
    np.testing.assert_allclose(result.decode()[0], result.control_points[0], atol=1e-12)


def test_span_support_and_overlap():
    d1 = map_delay(2)
    d2 = map_delay(4)
    within = map_delay(1)
    assert d1.support_control_indices == (0, 1, 2, 3)
    assert d2.support_control_indices == (0, 1, 2, 3, 4)
    assert set(d1.support_control_indices) & set(range(1, 5)) == {1, 2, 3}
    assert within.phase_steps == 1 and within.committed_control_count == 4
    assert within.affected_spans == 1
    assert map_delay(3).phase_steps == 1 and map_delay(3).committed_control_count == 5
    assert map_delay(10).committed_control_count == 8
    assert map_delay(0).committed_control_count == 0


def test_control_mask_uses_d_plus_three_controls_and_zero_is_empty():
    delays = torch.tensor([0, 1, 5])
    mask = control_support_mask(delays)
    assert mask.shape == (3, 18, 7)
    assert mask[0].sum() == 0
    assert mask[1].sum() == 4 * 7
    assert mask[2].sum() == 8 * 7


def test_hard_mask_exactly_matches_authoritative_basis_support():
    adapter = BSplineAdapter(Config())
    basis = adapter.basis
    for affected_spans in range(1, 6):
        raw_rows = affected_spans * adapter.config.span_length_steps
        authoritative_support = np.any(np.abs(basis[:raw_rows]) > 1e-12, axis=0)
        mask = control_support_mask(
            torch.tensor([affected_spans]),
            num_basis=adapter.config.num_basis,
            action_dim=1,
            degree=adapter.config.degree,
        )[0, :, 0].numpy()
        np.testing.assert_array_equal(mask, authoritative_support)


def test_raw_action_codec_shift_quantization_and_prefix(tmp_path):
    (tmp_path/"encoder.json").write_text('{"config":{"action_horizon":30},"calibration":{"low":[-1,-1,-1,-1,-1,-1,-1],"high":[1,1,1,1,1,1,1]}}')
    codec=RawActionCodec(tmp_path,torch.device("cpu"))
    actions=torch.linspace(-1,1,30*7).reshape(1,30,7)
    tokens=codec.encode_tokens(actions)
    assert tokens.shape==(1,30,7)
    assert (codec.decode_tokens(tokens)-actions).abs().max() <= 1/255+1e-6
    shifted=codec.shift_and_refit(actions,torch.tensor([3]))
    torch.testing.assert_close(shifted[:,:27],actions[:,3:])
    torch.testing.assert_close(shifted[:,27:],actions[:,-1:].expand(-1,3,-1))
    mask=raw_action_prefix_mask(torch.tensor([1,10]),30)
    assert mask.shape==(2,30,7) and mask[0].sum()==7 and mask[1].sum()==70


def test_raw_ttrtc_training_condition_uses_current_ground_truth_and_allows_zero_delay():
    target = torch.randn(2, 30, 7)
    batch = {"continuous_target": target}
    codec = SimpleNamespace(representation="raw", action_horizon=30)
    condition = make_reference_ttrtc_condition(
        batch, codec, torch.tensor([0, 3], dtype=torch.long)
    )
    assert torch.equal(condition["prefix_values"], target)
    assert not condition["fixed_mask"][0].any()
    assert condition["fixed_mask"][1, :3].all()
    assert not condition["fixed_mask"][1, 3:].any()


def test_discrete_zero_delay_uses_no_parent_history_or_fixed_rows():
    batch = {
        "previous_vision_features": torch.randn(2, 5, 2, 16, 8),
        "previous_state": torch.randn(2, 5, 7),
        "previous_parent_prediction": torch.randint(0, 256, (2, 5, 30, 7)),
        "has_previous": torch.ones(2, 5, dtype=torch.bool),
    }
    delays = torch.tensor([0, 3], dtype=torch.long)
    previous, available = _previous_batch(batch, delays)
    cached = _cached_parent_prediction(batch, delays)
    assert previous["vision_features"].shape == (2, 2, 16, 8)
    assert available.tolist() == [False, True]
    torch.testing.assert_close(cached[0], batch["previous_parent_prediction"][0, 0])
    torch.testing.assert_close(cached[1], batch["previous_parent_prediction"][1, 2])


def test_discrete_ttrtc_delay_laws_include_unconditional_zero_case():
    cfg = Config()
    cfg.policy.architecture = "discrete_joint"
    for representation, maximum in (("raw", 10), ("bspline", 5)):
        cfg.data.action_representation = representation
        torch.manual_seed(7)
        delays = _sample_training_delays(cfg, 4096, torch.device("cpu"))
        assert int(delays.min()) == 0
        assert int(delays.max()) <= maximum
        assert torch.any(delays > 0)


def test_binary_hard_mask_pigdm_has_no_effect_outside_conditioned_coordinates():
    noise = torch.tensor([[[0.0], [1.0], [2.0]]])
    values = torch.tensor([[[4.0], [99.0], [-99.0]]])
    mask = torch.tensor([[[True], [False], [False]]])

    result = hard_mask_pigdm_sample(
        noise,
        lambda state, time: torch.zeros_like(state),
        values,
        mask,
        steps=1,
        max_guidance_weight=1.0,
    )

    torch.testing.assert_close(result[mask], values[mask])
    torch.testing.assert_close(result[~mask], noise[~mask])


def test_binary_prefix_pigdm_matches_reference_equation_for_linear_velocity():
    noise = torch.tensor([[[0.3], [-0.4], [0.7]]], dtype=torch.float64)
    values = torch.tensor([[[1.2], [0.5], [-0.8]]], dtype=torch.float64)
    mask = torch.tensor([[[True], [True], [False]]])
    steps = 4
    maximum = 5.0
    slope = 0.2

    def velocity(state, time):
        return slope * state + time[:, None, None] * 0.1

    actual = hard_mask_pigdm_sample(
        noise,
        velocity,
        values,
        mask,
        steps=steps,
        max_guidance_weight=maximum,
    )

    # Independent scalar transcription of Kinetix FlowPolicy.realtime_action:
    # x1 = x_t + (1-t)v_t; correction = J_x(x1)^T (y-x1); then Euler step.
    expected = noise.clone()
    delta = 1.0 / steps
    for index in range(steps):
        time_value = index / steps
        velocity_value = slope * expected + time_value * 0.1
        endpoint = expected + (1.0 - time_value) * velocity_value
        endpoint_jacobian = 1.0 + (1.0 - time_value) * slope
        error = torch.where(mask, values - endpoint, torch.zeros_like(endpoint))
        correction = endpoint_jacobian * error
        if time_value == 0.0:
            guidance = maximum
        else:
            remaining = 1.0 - time_value
            inverse_r2 = (time_value**2 + remaining**2) / remaining**2
            guidance = min((remaining / time_value) * inverse_r2, maximum)
        expected = expected + delta * (velocity_value + guidance * correction)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_pigdm_trace_exposes_exact_per_step_guidance_counterfactual():
    noise = torch.tensor([[[0.2], [-0.3]]], dtype=torch.float64)
    values = torch.tensor([[[0.8], [4.0]]], dtype=torch.float64)
    mask = torch.tensor([[[True], [False]]])

    result, trace = hard_mask_pigdm_sample(
        noise,
        lambda state, time: 0.1 * state,
        values,
        mask,
        steps=3,
        max_guidance_weight=2.0,
        return_trace=True,
    )

    assert len(trace) == 3
    assert trace[0]["step"] == 0
    assert trace[0]["guidance_weight"] == 2.0
    for item in trace:
        expected = item["unguided_next"] + (
            item["guidance_weight"] / 3.0
        ) * item["vjp_correction"]
        torch.testing.assert_close(item["guided_next"], expected)
    torch.testing.assert_close(result, trace[-1]["guided_next"])


def test_pigdm_with_empty_prefix_matches_the_standard_euler_sampler():
    noise = torch.randn(2, 4, 3, dtype=torch.float64)
    values = torch.randn_like(noise)
    mask = torch.zeros_like(noise, dtype=torch.bool)
    steps = 6

    def velocity(state, time):
        return 0.15 * state + time[:, None, None] * 0.2

    guided = hard_mask_pigdm_sample(
        noise,
        velocity,
        values,
        mask,
        steps=steps,
    )
    standard = noise.clone()
    for index in range(steps):
        time = standard.new_full((len(standard),), index / steps)
        standard = standard + velocity(standard, time) / steps

    torch.testing.assert_close(guided, standard, rtol=0, atol=1e-15)
