
from numpy import *
import numpy as np

import numba
from numba import jit, jit_module, generated_jit, vectorize


def pearsonr(x:ndarray, y:ndarray):
    r"""
    +
        Pearson correlation coefficient and p-value for testing non-correlation.

        The Pearson correlation coefficient [1]_ measures the linear relationship
        between two datasets.  The calculation of the p-value relies on the
        assumption that each dataset is normally distributed.  (See Kowalski [3]_
        for a discussion of the effects of non-normality of the input on the
        distribution of the correlation coefficient.)  Like other correlation
        coefficients, this one varies between -1 and +1 with 0 implying no
        correlation. Correlations of -1 or +1 imply an exact linear relationship.

        Parameters
        ----------
        x : (N,) array_like
            Input array.
        y : (N,) array_like
            Input array.

        Returns
        -------
        r : float
            Pearson's correlation coefficient.
        p-value : float
            Two-tailed p-value.

        Warns
        -----
        PearsonRConstantInputWarning
            Raised if an input is a constant array.  The correlation coefficient
            is not defined in this case, so ``np.nan`` is returned.

        PearsonRNearConstantInputWarning
            Raised if an input is "nearly" constant.  The array ``x`` is considered
            nearly constant if ``norm(x - mean(x)) < 1e-13 * abs(mean(x))``.
            Numerical errors in the calculation ``x - mean(x)`` in this case might
            result in an inaccurate calculation of r.

        See Also
        --------
        spearmanr : Spearman rank-order correlation coefficient.
        kendalltau : Kendall's tau, a correlation measure for ordinal data.

        Notes
        -----
        The correlation coefficient is calculated as follows:

        .. math::

            r = \frac{\sum (x - m_x) (y - m_y)}
                    {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

        where :math:`m_x` is the mean of the vector x and :math:`m_y` is
        the mean of the vector y.

        Under the assumption that x and y are drawn from
        independent normal distributions (so the population correlation coefficient
        is 0), the probability density function of the sample correlation
        coefficient r is ([1]_, [2]_):

        .. math::

            f(r) = \frac{{(1-r^2)}^{n/2-2}}{\mathrm{B}(\frac{1}{2},\frac{n}{2}-1)}

        where n is the number of samples, and B is the beta function.  This
        is sometimes referred to as the exact distribution of r.  This is
        the distribution that is used in `pearsonr` to compute the p-value.
        The distribution is a beta distribution on the interval [-1, 1],
        with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
        implementation of the beta distribution, the distribution of r is::

            dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

        The p-value returned by `pearsonr` is a two-sided p-value. The p-value
        roughly indicates the probability of an uncorrelated system
        producing datasets that have a Pearson correlation at least as extreme
        as the one computed from these datasets. More precisely, for a
        given sample with correlation coefficient r, the p-value is
        the probability that abs(r') of a random sample x' and y' drawn from
        the population with zero correlation would be greater than or equal
        to abs(r). In terms of the object ``dist`` shown above, the p-value
        for a given r and length n can be computed as::

            p = 2*dist.cdf(-abs(r))

        When n is 2, the above continuous distribution is not well-defined.
        One can interpret the limit of the beta distribution as the shape
        parameters a and b approach a = b = 0 as a discrete distribution with
        equal probability masses at r = 1 and r = -1.  More directly, one
        can observe that, given the data x = [x1, x2] and y = [y1, y2], and
        assuming x1 != x2 and y1 != y2, the only possible values for r are 1
        and -1.  Because abs(r') for any sample x' and y' with length 2 will
        be 1, the two-sided p-value for a sample of length 2 is always 1.

        References
        ----------
        .. [1] "Pearson correlation coefficient", Wikipedia,
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        .. [2] Student, "Probable error of a correlation coefficient",
            Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
        .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
            of the Sample Product-Moment Correlation Coefficient"
            Journal of the Royal Statistical Society. Series C (Applied
            Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

        Examples
        --------
        >>> from scipy import stats
        >>> stats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
        (-0.7426106572325057, 0.1505558088534455)

        There is a linear dependence between x and y if y = a + b*x + e, where
        a,b are constants and e is a random error term, assumed to be independent
        of x. For simplicity, assume that x is standard normal, a=0, b=1 and let
        e follow a normal distribution with mean zero and standard deviation s>0.

        >>> s = 0.5
        >>> x = stats.norm.rvs(size=500)
        >>> e = stats.norm.rvs(scale=s, size=500)
        >>> y = x + e
        >>> stats.pearsonr(x, y)
        (0.9029601878969703, 8.428978827629898e-185) # may vary

        This should be close to the exact value given by

        >>> 1/np.sqrt(1 + s**2)
        0.8944271909999159

        For s=0.5, we observe a high level of correlation. In general, a large
        variance of the noise reduces the correlation, while the correlation
        approaches one as the variance of the error goes to zero.

        It is important to keep in mind that no correlation does not imply
        independence unless (x, y) is jointly normal. Correlation can even be zero
        when there is a very simple dependence structure: if X follows a
        standard normal distribution, let y = abs(x). Note that the correlation
        between x and y is zero. Indeed, since the expectation of x is zero,
        cov(x, y) = E[x*y]. By definition, this equals E[x*abs(x)] which is zero
        by symmetry. The following lines of code illustrate this observation:

        >>> y = np.abs(x)
        >>> stats.pearsonr(x, y)
        (-0.016172891856853524, 0.7182823678751942) # may vary

        A non-zero correlation coefficient can be misleading. For example, if X has
        a standard normal distribution, define y = x if x < 0 and y = 0 otherwise.
        A simple calculation shows that corr(x, y) = sqrt(2/Pi) = 0.797...,
        implying a high level of correlation:

        >>> y = np.where(x < 0, x, 0)
        >>> stats.pearsonr(x, y)
        (0.8537091583771509, 3.183461621422181e-143) # may vary

        This is unintuitive since there is no dependence of x and y if x is larger
        than zero which happens in about half of the cases if we sample x and y.
    """
    n = len(x)
    if n != len(y):
        raise ValueError('x and y must have the same length.')

    if n < 2:
        raise ValueError('x and y must have length at least 2.')

    x = np.asarray(x)
    y = np.asarray(y)

    # If an input is constant, the correlation coefficient is not defined.
    if (x == x[0]).all() or (y == y[0]).all():
        warnings.warn(PearsonRConstantInputWarning())
        return np.nan, np.nan

    # dtype is the data type for the calculations.  This expression ensures
    # that the data type is at least 64 bit floating point.  It might have
    # more precision if the input is, for example, np.longdouble.
    dtype = type(1.0 + x[0] + y[0])

    if n == 2:
        return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0])), 1.0

    xmean = x.mean(dtype=dtype)
    ymean = y.mean(dtype=dtype)

    # By using `astype(dtype)`, we ensure that the intermediate calculations
    # use at least 64 bit floating point.
    xm = x.astype(dtype) - xmean
    ym = y.astype(dtype) - ymean

    # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
    # scipy.linalg.norm(xm) does not overflow if xm is, for example,
    # [-5e210, 5e210, 3e200, -3e200]
    normxm = linalg.norm(xm)
    normym = linalg.norm(ym)

    threshold = 1e-13
    if normxm < threshold*abs(xmean) or normym < threshold*abs(ymean):
        # If all the values in x (likewise y) are very close to the mean,
        # the loss of precision that occurs in the subtraction xm = x - xmean
        # might result in large errors in r.
        warnings.warn(PearsonRNearConstantInputWarning())

    r = np.dot(xm/normxm, ym/normym)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    r = max(min(r, 1.0), -1.0)

    # As explained in the docstring, the p-value can be computed as
    #     p = 2*dist.cdf(-abs(r))
    # where dist is the beta distribution on [-1, 1] with shape parameters
    # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
    # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
    # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
    # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
    # to avoid a TypeError raised by btdtr when r is higher precision.)
    ab = n/2 - 1
    prob = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(r))))

    return r, prob

