"""
Utilities for stats.

Author: Sam Cohan
"""

import numpy as np
import pandas as pd
from scipy import stats


def weighted_percentile(vals, percentiles, weights=None, vals_sorted=False):
    """This function is very close to to np.percentile, but supports weights.

    Args
        val (ArrayLike): array-like of values
        percentiles (ArrayLike): array-like of percentiles in range [0,100]
        weights (ArrayLike): array-like of the same length as vals (defaults
            to None meaning equal weights)
        vals_sorted (bool): boolean indicating whether vals are already
            sorted (defaults to False, and will sort the values)
    """
    if weights is None:
        return np.percentile(vals, percentiles)
    vals = np.array(vals)
    quantiles = np.array(percentiles) / 100.0
    weights = np.array(weights)

    if not vals_sorted:
        sorter = np.argsort(vals)
        vals = vals[sorter]
        weights = weights[sorter]

    weighted_quantiles = np.cumsum(weights) - 0.5 * weights
    # Make sure results match np.percentile
    weighted_quantiles -= weighted_quantiles[0]
    weighted_quantiles /= weighted_quantiles[-1]
    return np.interp(quantiles, weighted_quantiles, vals)


def get_pctl_range_func(lower_pctl, upper_pctl):
    """Generate named function for calculating inter-quartile ranges.

    This comes in handy for creating named functions to be used with pandas
    groupby aggregate. The function name will follow pattern
    pctl_range_<two_digit_lower_pctl>_two_digit_upper_pctl.

    Args:
        lower_pctl (int): int between 0 and 99 indicating the lower
            percentile score.
        upper_pctl (int): int between upper_pctl and 99 indicating the upper
            percentile score.
    """

    def pctl_range_func(x):
        if len(x) == 0 or np.all(np.isnan(x)):  # pylint: disable=len-as-condition
            return np.NaN
        pctl_vals = np.nanpercentile(x, [lower_pctl, upper_pctl])
        return pctl_vals[1] - pctl_vals[0]

    pctl_range_func.__name__ = "pctl_range_%02.0f_%02.0f" % (lower_pctl, upper_pctl)
    pctl_range_func.__doc__ = (
        "Returns the difference between %02.0f and %02.0f percentile values."
        % (upper_pctl, lower_pctl)
    )
    return pctl_range_func


def get_pctl_func(pctl, drop_zeros=False):
    """Generate named function for calculating a percentile value.

    This comes in handy for creating named functions to be used with pandas
    groupby aggregate.
    The function name will follow pattern pctl_<two_digit_pctl_score>.

    Args:
        pctl (int): int between 0 and 100 indicating the percentile score to
            calculate value for.
        drop_zero -- boolean indicating whether zero values should be dropped
            before calculating the percentile (note that nulls are always
            dropped regardless)
    """

    def pctl_func(x):
        if drop_zeros:
            x = x[~pd.isnull(x) & (x != 0)]
        else:
            x = x[~pd.isnull(x)]
        if not len(x):  # pylint: disable=len-as-condition
            return np.NaN
        return np.percentile(x, pctl)

    pctl_func.__name__ = "{}pctl_{:02.0f}".format("nz_" if drop_zeros else "", pctl)
    pctl_func.__doc__ = "Returns the {}{:02.0f}-th percentile".format(
        "non-zero " if drop_zeros else "", pctl
    )

    return pctl_func


pctl_05 = get_pctl_func(5)
pctl_10 = get_pctl_func(10)
pctl_50 = get_pctl_func(50)
pctl_90 = get_pctl_func(90)
pctl_95 = get_pctl_func(95)


def nz_mean(x):
    """Calcualte non-zero mean of the input (nans are also dropped)"""
    if not isinstance(x, (pd.Series, np.ndarray)):
        x = np.array(x)
    x = x[~pd.isnull(x) & (x != 0)]
    if not len(x):  # pylint: disable=len-as-condition
        return np.NaN
    return x.mean()


def get_anomalous_prob(x_pos, n_trial, non_anomalous_interval):
    """Get probability that x positive outcomes out of n trails comes from a
    process with probability within the provided non-anomalous interval.

    Uses two applications of one-sided binomial test to figure out the
    complement of the probability that the observation falls within the
    non-anomalous interval. Note that if the n_pos is close to 0, we use the
    right tail test (i.e. hypothesis of being greater than the interval
    bounds with alternative being less) because if we see zero events, the
    binomial test thinks there is zero chance of being greater than zero.
    Conversely, if n_pos is close to 100%, we use the left tail test (i.e.
    hypothesis of being less than interval bounds with alternative being
    greater) because if we see every single event as positive, the test
    assumes there is no chance of an event not converting.

    Args:
        x_pos (int): number of positive outcomes
        n_trial (int): integer number of trials
        non_anomalous_interval (Tuple[float, float]) -- tuple of two floats
        indicating lower and upper bounds outside of which is considered anomalous.
    """
    if n_trial < 2 or x_pos > n_trial:
        return np.NaN
    # use hypothesis that n_pos/n_trial is greater than the interval bounds
    alternative = "less"
    if (float(x_pos) / n_trial) > 0.5:
        alternative = (
            "greater"  # use hypothesis that n_pos/n_trial is less than interval bounds
        )
    return 1 - abs(
        stats.binom_test(
            x=x_pos, n=n_trial, p=non_anomalous_interval[0], alternative=alternative
        )
        - stats.binom_test(
            x=x_pos, n=n_trial, p=non_anomalous_interval[1], alternative=alternative
        )
    )


def get_binomial_conf_interval(x_pos, n_tot, conf=0.95):
    """Returns the Agresti-Couli Interval ([lower-bound upper-bound]).
    See
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Agresti-Coull_Interval
    A
    good discussion of the various confidence interval formulae is in this paper:
    http://projecteuclid.org/DPubS/Repository/1.0/Disseminate?view=body&id=pdf_1&handle=euclid.ss/1009213286;
    it is shown in particular that the A-C interval always strictly contains
    the Wilson interval, and that A-C is a good choice for n > 40. As we are
    almost always dealing with large n and would prefer to have more conservative
    confidence interval estimates, A-C seems a good fit.

    Args
        x_pos (int): natural integer indicating number of positive samples
        n_tot (int): natural integer indicating number of total samples
        conf (float): float in range [0, 1] indicating probability of success
        (defaults to 0.95)
    """
    from scipy.stats import norm

    assert x_pos <= n_tot, "cannot have x_pos={} > n_tot={}".format(x_pos, n_tot)
    alpha = 1 - conf
    z = norm.ppf(1 - alpha / 2)
    nt = n_tot + z ** 2  # n_tilde in the formula
    pt = (x_pos + z ** 2 / 2) / nt  # p_tilde in the formula
    hw = z * np.sqrt(pt * (1 - pt) / nt)
    return (pt - hw), (pt + hw)
