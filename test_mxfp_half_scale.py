import torch

from quant.ops.mxfp import _quantize_mx, _shared_exponents


def test_floor_is_the_reference_exponent():
    x = torch.full((1, 16), 3.8)
    expected = torch.tensor([[1.0]])
    for mode in (None, 0):
        torch.testing.assert_close(_shared_exponents(x, axes=[1], minus_exp=mode), expected)
    torch.testing.assert_close(_shared_exponents(x, axes=[1], minus_exp=1), expected - 1)


def test_auto_selects_lower_mse_per_block():
    x = torch.tensor([[2.0] + [0.13] * 63, [2.0] + [1.0] * 63])

    def quantize(mode):
        return _quantize_mx(x, 8, "fp4_e2m1", axes=-1, block_size=64, minus_exp=mode)

    baseline, half, auto = quantize(None), quantize(1), quantize("auto")
    baseline_mse = (x - baseline).square().mean(dim=-1)
    half_mse = (x - half).square().mean(dim=-1)
    assert half_mse[0] < baseline_mse[0]
    assert baseline_mse[1] < half_mse[1]
    torch.testing.assert_close(auto[0], half[0])
    torch.testing.assert_close(auto[1], baseline[1])


def test_auto_selects_lower_mse_for_16_element_blocks():
    torch.manual_seed(7)
    x = torch.randn(3, 35)
    baseline = _quantize_mx(x, 8, "fp4_e2m1", axes=-1, block_size=16)
    half = _quantize_mx(x, 8, "fp4_e2m1", axes=-1, block_size=16, minus_exp=1)
    auto = _quantize_mx(x, 8, "fp4_e2m1", axes=-1, block_size=16, minus_exp="auto")
    baseline_mse = (x - baseline).square().reshape(3, -1)
    half_mse = (x - half).square().reshape(3, -1)
    for start in (0, 16, 32):
        stop = min(start + 16, x.shape[-1])
        for row in range(x.shape[0]):
            row_expected = half if half_mse[row, start:stop].mean() < baseline_mse[row, start:stop].mean() else baseline
            torch.testing.assert_close(auto[row, start:stop], row_expected[row, start:stop])


if __name__ == "__main__":
    test_floor_is_the_reference_exponent()
    test_auto_selects_lower_mse_per_block()
    test_auto_selects_lower_mse_for_16_element_blocks()


"""PYTHONPATH=/mnt/lm_data_afs/charles/codes/LightX2V/fake_quant_ops /mnt/lm_data_afs/charles/codes/LightX2V/.venv/bin/python /mnt/lm_data_afs/charles/codes/LightX2V/fake_quant_ops/test_mxfp_half_scale.py"""
