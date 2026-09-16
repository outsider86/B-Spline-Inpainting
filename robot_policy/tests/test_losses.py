import torch

from robot_policy.policies.common import expected_token_distance


def _loss(probabilities, target=100):
    logits = probabilities.log().reshape(1, 1, 256)
    y = torch.tensor([[target]])
    valid = torch.ones_like(y, dtype=torch.bool)
    return expected_token_distance(logits, y, valid)[1]


def test_expected_distance_acceptance_cases():
    exact = torch.full((256,), 1e-12); exact[100] = 1
    near = torch.full((256,), 1e-12); near[100] = .5; near[101] = .5
    far = torch.full((256,), 1e-12); far[100] = .5; far[110] = .5
    symmetric = torch.full((256,), 1e-12); symmetric[90] = .5; symmetric[110] = .5
    assert _loss(exact) < 1e-6
    assert _loss(far) > _loss(near)
    assert _loss(symmetric) > 9.9


def test_expected_distance_gradient_and_mask():
    logits = torch.randn(2, 3, 256, requires_grad=True)
    target = torch.tensor([[1, 2, 3], [4, 5, 6]])
    valid = torch.tensor([[True, False, True], [False, True, False]])
    ce, l1 = expected_token_distance(logits, target, valid)
    (ce + l1).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[~valid].abs().sum() == 0
    assert logits.grad[valid].abs().sum() > 0

