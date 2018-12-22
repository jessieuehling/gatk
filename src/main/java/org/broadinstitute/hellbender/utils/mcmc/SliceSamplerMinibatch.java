package org.broadinstitute.hellbender.utils.mcmc;


import org.apache.commons.math3.distribution.ExponentialDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Implements slice sampling of a continuous, univariate, unnormalized probability density function,
 * which is assumed to be unimodal.  See Neal 2003 at https://projecteuclid.org/euclid.aos/1056562461 for details.
 * Optional minibatching is implemented as in http://proceedings.mlr.press/v33/dubois14.pdf.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class SliceSamplerMinibatch extends SliceSampler {
    private final Integer minibatchSize;
    private final Double minibatchApproxThreshold;

    /**
     * Creates a new sampler, given a random number generator, a continuous, univariate, unimodal, unnormalized
     * log probability density function, hard limits on the random variable, a step width, a minibatch size,
     * and a minibatch approximation threshold.
     * @param rng                       random number generator
     * @param logPDF                    continuous, univariate, unimodal log probability density function (up to additive constant)
     * @param xMin                      minimum allowed value of the random variable
     * @param xMax                      maximum allowed value of the random variable
     * @param width                     step width for slice expansion
     * @param minibatchSize             minibatch size
     * @param minibatchApproxThreshold  minibatchApproxThreshold size
     */
    public SliceSamplerMinibatch(final RandomGenerator rng, final Function<Double, Double> logPDF,
                                 final double xMin, final double xMax, final double width, final int minibatchSize, final double minibatchApproxThreshold) {
        super(rng, logPDF, xMin, xMax, width);
        ParamUtils.isPositive(minibatchSize, "Minibatch size must be positive.");
        ParamUtils.isPositive(minibatchApproxThreshold, "Minibatch approximation threshold must be positive.");
        this.minibatchSize = minibatchSize;
        this.minibatchApproxThreshold = minibatchApproxThreshold;
    }

    /**
     * Creates a new sampler, given a random number generator, a continuous, univariate, unimodal, unnormalized
     * log probability density function, a step width, a minibatch size, and a minibatch approximation threshold.
     * @param rng      random number generator
     * @param logPDF   continuous, univariate, unimodal log probability density function (up to additive constant)
     * @param width    step width for slice expansion
     * @param minibatchSize             minibatch size
     * @param minibatchApproxThreshold  minibatchApproxThreshold size
     */
    public SliceSamplerMinibatch(final RandomGenerator rng, final Function<Double, Double> logPDF, final double width, final int minibatchSize, final double minibatchApproxThreshold) {
        this(rng, logPDF, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, width, minibatchSize, minibatchApproxThreshold);
    }

    /**
     * Generate a single sample from the probability density function, given an initial value to use in slice construction.
     * @param xInitial      initial value to use in slice construction; must be in [xMin, xMax]
     * @return              sample drawn from the probability density function
     */
    public double sample(final double xInitial) {
        Utils.validateArg(xMin <= xInitial && xInitial <= xMax, "Initial point in slice sampler is not within specified range.");

        //adjust xInitial if on boundary
        final double xSample = Math.min(Math.max(xInitial, xMin + EPSILON), xMax - EPSILON);

        //follow Neal 2003 procedure to slice sample with doubling of interval (assuming unimodal distribution)

        //randomly position slice with given width so that it brackets xSample; position is uniformly distributed
        double xLeft = xSample - width * rng.nextDouble();
        double xRight = xLeft + width;

        //sample the variable used to randomly pick height of slice from uniform distribution under PDF(xSample);
        //slice height = u * PDF(xSample), where u ~ Uniform(0, 1)
        //however, since we are working with logPDF, we instead sample z = -log u ~ Exponential(1)
        final double z = exponentialDistribution.sample();

        int k = MAXIMUM_NUMBER_OF_DOUBLINGS;
        //expand slice interval by doubling until it brackets PDF
        //(i.e., PDF at both ends is less than the slice height)
        while (k > 0 && (isOverSliceHeight(xLeft, xSample, z) || isOverSliceHeight(xRight, xSample, z))) {
            if (rng.nextBoolean()) {
                xLeft = xLeft - (xRight - xLeft);
            } else {
                xRight = xRight + (xRight - xLeft);
            }
            k--;
        }

        //sample uniformly from slice interval until sample over slice height found, shrink slice on each iteration if not found
        //limited to MAXIMUM_NUMBER_OF_SLICE_SAMPLINGS, after which last proposed sample is returned
        //(shouldn't happen if width is chosen appropriately)
        int numIterations = 1;
        double xProposed = rng.nextDouble() * (xRight - xLeft) + xLeft;
        while (numIterations <= MAXIMUM_NUMBER_OF_SLICE_SAMPLINGS) {
            if (isOverSliceHeight(xProposed, xSample, z)) {
                break;
            }
            if (xProposed < xSample) {
                xLeft = xProposed;
            } else {
                xRight = xProposed;
            }
            xProposed = rng.nextDouble() * (xRight - xLeft) + xLeft;
            numIterations++;
        }
        return Math.min(Math.max(xProposed, xMin + EPSILON), xMax - EPSILON);
    }

    /**
     * Generate multiple samples from the probability density function, given an initial value to use in slice construction.
     * @param xInitial      initial value to use in slice construction; if outside [xMin, xMax], forced to be within
     * @param numSamples    number of samples to generate
     * @return              samples drawn from the probability density function
     */
    public List<Double> sample(final double xInitial, final int numSamples) {
        ParamUtils.isPositive(numSamples, "Number of samples must be positive.");
        final List<Double> samples = new ArrayList<>(numSamples);
        double xSample = xInitial;
        for (int i = 0; i < numSamples; i++) {
            xSample = sample(xSample);
            samples.add(xSample);
        }
        return samples;
    }

    /**
     * Returns true if PDF(xProposed) is over slice height = u * PDF(xSample), where u = exp(-z).
     */
    private boolean isOverSliceHeight(final double xProposed,
                                      final double xSample,
                                      final double z) {
        if (xProposed < xMin || xMax < xProposed) {
            return false;
        }
        if (xSampleCache == null || xSampleCache != xSample) {
            //we cache logPDF(xSample) to avoid evaluating it multiple times
            xSampleCache = xSample;
            logPDFCache = logPDF.apply(xSample);
        }
        if ((xSampleCache == null && logPDFCache != null) || (xSampleCache != null && logPDFCache == null)) {
            throw new GATKException.ShouldNeverReachHereException("Cache for logPDF(xSample) is invalid.");
        }
        return logPDF.apply(xProposed) > logPDFCache - z;
    }
}