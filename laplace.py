import torch
import numpy as np
from scipy import stats

def laplace_gof_test(
    tensor,
    assume_zero_mean=True,
    subsample=500_000,
    seed=0
):
    """
    Goodness-of-fit test for Laplace(0, b).

    Args:
        tensor (torch.Tensor): input tensor
        assume_zero_mean (bool): fix mu=0 or estimate mu
        subsample (int): max number of samples for KS test
    """
    x = tensor.detach().float().cpu().numpy().reshape(-1)

    # Subsampling for KS stability
    if x.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        x = rng.choice(x, size=subsample, replace=False)

    # Parameter estimation
    if assume_zero_mean:
        mu_hat = 0.0
        b_hat = np.mean(np.abs(x))
    else:
        mu_hat = np.median(x)
        b_hat = np.mean(np.abs(x - mu_hat))

    # KS test against Laplace
    ks_stat, p_value = stats.kstest(
        x,
        'laplace',
        args=(mu_hat, b_hat)
    )

    # Moments (diagnostic only)
    mean = np.mean(x)
    std = np.std(x)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x, fisher=False)  # Laplace => 6

    return {
        "mu_hat": mu_hat,
        "b_hat": b_hat,
        "std": std,
        "mean": mean,
        "skew": skew,
        "kurtosis": kurt,
        "ks_stat": ks_stat,
        "p_value": p_value,
        "num_samples": x.shape[0],
    }
fwd_tensor = torch.load('data/real/fwd_in_140541200265632.pt').cuda()
bwd_tensor = torch.load('data/real/bwd_quant_temp_140541200614960.pt').cuda()
res_fwd = laplace_gof_test(fwd_tensor)
res_bwd = laplace_gof_test(bwd_tensor)

print(res_fwd)
print(res_bwd)
