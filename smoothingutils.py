import math

import pandas as pd

from .logutils import setup_logger

LN_2 = math.log(2)

LOGGER = setup_logger(__name__)


class StreamExponentialSmoother:
    """Streaming Exponential Smoother.

    level = alpha * datum + (1 - alpha) * level_prev
    """

    def __init__(
        self,
        alpha=None,
        tau=None,
        halflife=None,
        span=None,
        unit_td=1,
        warn_if_time_rewinds=True,
        raise_if_time_rewinds=True,
    ):
        """Make an instance of StreamExponentialSmoother.

        Must provide exactly one of the parameters for the smoother.
        The exponential smoothing formula is as follows:
        ```
        level = alpha(td) * datum + (1-alpha(td)) * level_prev
        ```
        In the above, alpha is a function of timedelta `td` such that
        ```
        alpha(td) = 1 - exp(-td/tau)
        ```

        which implies:
        ```
        tau = td / ln(1 / (1-alpha))
        ```
        `tau` is the time-constant or the time it takes for a weight to
        decline to 1-1/e (~ 63.2%) of its original value. Veryy close related
        to tau is the `halflife` is the most intuitive parameter and is
        defined as time it takes for a weight to decay to half its previous
        value:
        ```
        1/2 = exp(-halflife/tau)
        --> 2 = exp(halflife/tau)
        --> halflife = ln(2) * tau
        ```
        If a constant `alpha` is provided, then an unit_td should also be
        provided with it to explicitely state the units of time between
        equally-spaced points (required for having tau in correct units).

        Arguments:
            alpha (float): alpha parameter of smoother (a.k.a smoothing factor)
                If you provide alpha, you should also provide `unit_td` which
                is the timedelta units between the equally spaced points.
            tau (float): time constant of smoother.
            halflife (float): the period of time for the exponential weight to
                reduce to one half. (alpha = 1 - exp(ln(1/2)/halflife))
            span (float): corresponds to what is commonly called an “N-day EW
                moving average”. (alpha = 2 / (span+1))
            unit_td (float): time units between equally spaced points. (i.e.
                if timedelta between two points is this, then the implied
                alpha will be equal to the constant alpha passed in.)
            raise_if_time_rewinds: whether an exception should be raised if
                timestamp of consumed data is less than the last consumed
                timestamp (defaults to True).
            warn_if_time_rewinds: whether a warning message should be logged
                if the timestamp of consumed data is less than the last consumed
                timestamp. (defaults to True).
            raise_if_time_rewinds: whether an exception should be raised if
                timestamp of consumed data is less than the last consumed
                timestamp. If False, the point that is back in time will just
                be ignored. (defaults to True).
        """
        assert unit_td > 0
        self._unit_td = unit_td
        if halflife:
            assert (not alpha) and (not tau) and (not span)
            assert halflife > 0
            self._halflife = halflife
            self._tau = self._halflife / LN_2
            self._alpha = 1 - math.exp(-unit_td / self._tau)
            self._span = (2. / self._alpha) - 1
        elif tau:
            assert (not alpha) and (not halflife) and (not span)
            assert tau > 0
            self._tau = tau
            self._halflife = self._tau * LN_2
            self._alpha = 1 - math.exp(-unit_td / self._tau)
            self._span = (2. / self._alpha) - 1
        elif alpha:
            assert (not tau) and (not halflife) and (not span)
            assert 0 < alpha < 1
            self._alpha = alpha
            self._tau = unit_td / math.log(1 / (1 - self._alpha))
            self._halflife = self._tau * LN_2
            self._span = (2. / self._alpha) - 1
        elif span:
            assert (not tau) and (not halflife) and (not alpha)
            assert span >= 1
            self._span = span
            self._alpha = 2. / (self._span + 1)
            self._tau = unit_td / math.log(1 / (1 - self._alpha))
            self._halflife = self._tau * LN_2
        else:
            raise Exception(
                "Provide exactly one of alpha, halflife, tau, or span.")
        self.warn_if_time_rewinds = warn_if_time_rewinds
        self.raise_if_time_rewinds = raise_if_time_rewinds
        self._level = None
        self._num_points = 0
        self._last_datum = None
        self._last_ts = None
        self._nans_in_a_row = 0

    def consume(self, datum, ts=None):
        """Feed next value in the stream to update internal state.

        Arguments:
            datum (float): next data point
            ts (num): current absolute time (default to None if you prefer
                to imply equally spaced points without specifying time unit.)

        Returns:
            (float): smoothed level
        """
        if pd.isnull(datum):
            self._nans_in_a_row += 1
            return datum
        if self._num_points < 1:
            self._level = datum
        else:  # self._num_point > 0
            td = None
            if ts is None:
                if self._last_ts is not None:
                    raise Exception(
                        "Cannot stop giving timestamps all of a sudden!")
                td = 1 + self._nans_in_a_row
            else:  # ts is not None
                if self._last_ts is None:
                    raise Exception(
                        "Cannot start giving timestamps all of a sudden!")
                # last_ts is not None
                if ts < self._last_ts:
                    error_msg = f"Stream going back in time: ts={ts} < last_ts={self._last_ts}"
                    if self.warn_if_time_rewinds:
                        LOGGER.warning(error_msg)
                    if self.raise_if_time_rewinds:
                        raise Exception(error_msg)
                    # If there is not exception, then we should just return the previous level.
                    return self._level
                td = ts - self._last_ts
            alpha = self.get_alpha(td)
            self._level = alpha * datum + (1 - alpha) * self._level

        self._num_points += 1
        self._last_datum = datum
        self._last_ts = ts
        self._nans_in_a_row = 0
        return self._level

    def get_alpha(self, td=1):
        return 1 - math.exp(-td / self._tau)

    @property
    def level(self):
        return self._level

    @property
    def last_ts(self):
        return self._last_ts

    @property
    def tau(self):
        return self._tau

    @property
    def halflife(self):
        return self._halflife

    @property
    def unit_td(self):
        return self._unit_td

    def reset_to(self, level, last_datum, last_ts, num_points):
        self._level = level
        self._last_datum = last_datum
        self._last_ts = last_ts
        self._num_points = num_points

    def __str__(self):
        return ("StreamExponentialSmoother("
                f"alpha={self.get_alpha(self.unit_td):,.4f}"
                f", halflife={self.halflife:,.4f}"
                f", num_points={self._num_points}"
                f", level={self.level}"
                f", last_ts={self.last_ts}"
                f", unit_td={self.unit_td}"
                ")")

    __repr__ = __str__


class StreamDoubleExponentialSmoother:
    """Streaming Double Exponential Smoother.

    level = alpha * datum + (1 - alpha) * (level_prev + trend_prev)
    trend = beta * (level - level_prev) + (1 - beta) * trend_prev
    forecast = level + trend

    """

    def __init__(
        self,
        alpha=None,
        beta=None,
        level_tau=None,
        trend_tau=None,
        level_halflife=None,
        trend_halflife=None,
        unit_td=1,
        warn_if_time_rewinds=True,
        raise_if_time_rewinds=True,
    ):
        """Make an instance of DoubleExponentialStreamSmoother.

        Arguments:
            alpha (float): alpha parameter of smoother (a.k.a smoothing factor)
                If you provide alpha, you should also provide `unit_td` which
                is the timedelta units between the equally spaced points.
            beta (float): beta parameter of smoother (a.k.a trend factor)
                If you provide beta, you should also provide `unit_td` which
                is the timedelta units between the equally spaced points.
            level_tau (float): time constant of level.
            trend_tau (float): time constant of trend.
            level_halflife (float): halflife of level.
            trend_halflife (float): halflife of trend.
            unit_td (float): time units between equally spaced points. (i.e.
                if timedelta between two points is this, then the implied
                alpha will be equal to the constant alpha passed in.)
            warn_if_time_rewinds: whether a warning message should be logged
                if the timestamp of consumed data is less than the last consumed
                timestamp. (defaults to True).
            raise_if_time_rewinds: whether an exception should be raised if
                timestamp of consumed data is less than the last consumed
                timestamp. If False, the point that is back in time will just
                be ignored. (defaults to True).
        """
        assert unit_td > 0
        self._unit_td = unit_td
        if level_halflife or trend_halflife:
            assert (not alpha) and (not beta)
            assert (not level_tau) and (not trend_tau)
            assert (level_halflife > 0) and (trend_halflife > 0)
            self._level_halflife = level_halflife
            self._trend_halflife = trend_halflife
            self._level_tau = self._level_halflife / LN_2
            self._trend_tau = self._trend_halflife / LN_2
            self._alpha = 1 - math.exp(-unit_td / self._level_tau)
            self._beta = 1 - math.exp(-unit_td / self._trend_tau)
        elif level_tau or trend_tau:
            assert (not alpha) and (not beta)
            assert (not level_halflife) and (not trend_halflife)
            assert (level_tau > 0) and (trend_tau > 0)
            self._level_tau = level_tau
            self._trend_tau = trend_tau
            self._level_halflife = self._level_tau * LN_2
            self._trend_halflife = self._trend_tau * LN_2
            self._alpha = 1 - math.exp(-unit_td / self._level_tau)
            self._beta = 1 - math.exp(-unit_td / self._trend_tau)
        elif alpha or beta:
            assert (not level_tau) and (not trend_tau)
            assert (not level_halflife) and (not trend_halflife)
            assert (0 < alpha < 1) and (0 < beta < 1)
            self._alpha = alpha
            self._beta = beta
            self._level_tau = unit_td / math.log(1 / (1 - alpha))
            self._trend_tau = unit_td / math.log(1 / (1 - beta))
            self._level_halflife = self._level_tau * LN_2
            self._tren_halflife = self._trend_tau * LN_2

        self.warn_if_time_rewinds = warn_if_time_rewinds
        self.raise_if_time_rewinds = raise_if_time_rewinds
        self._level = None
        self._trend = None
        self._num_points = 0
        self._last_datum = None
        self._first_ts = None
        self._last_ts = None
        self._nans_in_a_row = 0
        # While in _init_mode, will apply simple exponential smoothing.
        self._init_mode = True

    def consume(self, datum, ts=None):
        """Feed next value in the stream to update internal state.

        Arguments:
            datum (float): next data point
            ts (num): current absolute time (default to None if you prefer
                to imply equally spaced points without specifying time unit.)

        Returns:
            (float): smoothed level
        """
        if pd.isnull(datum):
            self._nans_in_a_row += 1
            return datum
        if self._num_points == 0:
            self._first_ts = ts
            self._level = datum
            self._trend = 0
        else:  # self._num_point > 0
            td = None
            if ts is None:
                if self._last_ts is not None:
                    raise Exception(
                        "Cannot stop giving timestamps all of a sudden!")
                td = 1 + self._nans_in_a_row
                # Make sure we have at least as many points as the halflife before we
                # start double exponentially smoothing. (Assumes that the points
                # are equally-spaced and one unit from each other).
                if self._num_points > self._level_halflife:
                    self._init_mode = False
            else:  # ts is not None
                if self._last_ts is None:
                    raise Exception(
                        "Cannot start giving timestamps all of a sudden!")
                # last_ts is not None
                if ts < self._last_ts:
                    error_msg = f"Stream going back in time: ts={ts} < last_ts={self._last_ts}"
                    if self.warn_if_time_rewinds:
                        LOGGER.warning(error_msg)
                    if self.raise_if_time_rewinds:
                        raise Exception(error_msg)
                    # If there is not exception, then we should just return the previous level.
                    return self._level
                td = ts - self._last_ts
                if self._init_mode:
                    time_since_first = ts - self._first_ts
                    # Make sure we have at least 5 points and are time_constant from start
                    # before we start double exponential smoothing.
                    if (self._num_points + 1) >= 5 and time_since_first > self._level_halflife:
                        self._init_mode = False

            alpha = self.get_alpha(td)
            beta = self.get_beta(td)
            # If in init_mode, will apply simple exponential smoothing to avoid having
            # crazy shakey start due to the trend factor being completely off.
            if self._init_mode:
                self._level = alpha * datum + (1 - alpha) * self._level
            else:
                _level = alpha * datum + \
                    (1 - alpha) * (self._level + self._trend)
                self._trend = beta * (_level - self._level) + \
                    (1 - beta) * self._trend
                self._level = _level

        self._num_points += 1
        self._last_datum = datum
        self._last_ts = ts
        self._nans_in_a_row = 0
        return self._level

    def get_forecast_for_timedelta(self, td):
        return self._level + td * self._trend

    def get_forecast_for_time(self, ts):
        if ts < self._last_ts:
            warn_msg = f"Asking for forecast back in time: ts={ts} < last_ts={self._last_ts}"
            LOGGER.warning(warn_msg)
            if self.raise_if_time_rewinds:
                raise Exception(warn_msg)
            return self._level
        return self._level + (ts - self._last_ts) * self._trend

    def get_alpha(self, td=1):
        return 1 - math.exp(-td / self._level_tau)

    def get_beta(self, td=1):
        return 1 - math.exp(-td / self._trend_tau)

    @property
    def level(self):
        return self._level

    @property
    def trend(self):
        return self._trend

    @property
    def last_ts(self):
        return self._last_ts

    @property
    def level_tau(self):
        return self._level_tau

    @property
    def trend_tau(self):
        return self._trend_tau

    @property
    def level_halflife(self):
        return self._level_halflife

    @property
    def trend_halflife(self):
        return self._trend_halflife

    @property
    def unit_td(self):
        return self._unit_td

    def __str__(self):
        return ("StreamDoubleExponentialSmoother("
                f"alpha={self.get_alpha(self.unit_td):,.4f}"
                f", beta={self.get_beta(self.unit_td):,.4f}"
                f", level_halflife={self.level_halflife:,.4f}"
                f", trend_halflife={self.trend_halflife:,.4f}"
                f", num_points={self._num_points}"
                f", level={self.level}"
                f", trend={self.trend}"
                f", last_ts={self.last_ts}"
                f", unit_td={self.unit_td}"
                ")")

    __repr__ = __str__


class ExpandingSmootherWrapper:
    def __init__(self, smoother):
        self.smoother = smoother
        assert hasattr(self.smoother, "consume")

    def __call__(self, data):
        return self.smoother.consume(data[-1])

    def __str__(self):
        return f"ExpandingSmootherWrapper({self.smoother})"
