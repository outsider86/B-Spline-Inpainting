import numpy as np
import torch

from robot_policy.config import Config
from robot_policy.encoders.bspline_adapter import BSplineAdapter
from robot_policy.rtc.delay_mapping import control_support_mask, map_delay, raw_action_prefix_mask
from robot_policy.rtc.training import RawActionCodec


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


def test_control_mask_uses_d_plus_three_controls():
    delays = torch.tensor([1, 5])
    mask = control_support_mask(delays)
    assert mask.shape == (2, 18, 7)
    assert mask[0].sum() == 4 * 7
    assert mask[1].sum() == 8 * 7


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
