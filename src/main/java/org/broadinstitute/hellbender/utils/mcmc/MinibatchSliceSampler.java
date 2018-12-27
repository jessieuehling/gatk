package org.broadinstitute.hellbender.utils.mcmc;


import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.primes.Primes;
import org.apache.commons.math3.random.RandomGenerator;
import org.broadinstitute.hellbender.exceptions.GATKException;
import org.broadinstitute.hellbender.utils.Utils;
import org.broadinstitute.hellbender.utils.param.ParamUtils;

import java.util.*;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Implements slice sampling of a continuous, univariate, unnormalized probability density function (PDF),
 * which is assumed to be unimodal.  See Neal 2003 at https://projecteuclid.org/euclid.aos/1056562461 for details.
 * Minibatching is implemented as in http://proceedings.mlr.press/v33/dubois14.pdf and requires
 * that the PDF, which is assumed to be a posterior function of a parameter value and the data, is specified in terms
 * of a prior, a likelihood, and the data.
 *
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
public final class MinibatchSliceSampler<DATA> extends AbstractSliceSampler {
    private final List<DATA> data;
    private final Function<Double, Double> logPrior;
    private final BiFunction<DATA, Double, Double> logLikelihood;
    private final Integer minibatchSize;
    private final Double approxThreshold;

    private final int numDataPoints;

    private Double xSampleCache = null;
    private Double logPriorCache = null;
    private Map<Integer, Double> logLikelihoodsCache = null;    //data index -> log likelihood

    /**
     * Creates a new sampler for a bounded univariate random variable, given a random number generator, a list of data,
     * a continuous, univariate, unimodal, unnormalized log probability density function
     * (assumed to be a posterior and specified by a prior and a likelihood),
     * hard limits on the random variable, a step width, a minibatch size, and a minibatch approximation threshold.
     * @param rng                       random number generator
     * @param data                      list of data
     * @param logPrior                  log prior component of continuous, univariate, unimodal log posterior (up to additive constant)
     * @param logLikelihood             log likelihood component of continuous, univariate, unimodal log posterior (up to additive constant)
     * @param xMin                      minimum allowed value of the random variable
     * @param xMax                      maximum allowed value of the random variable
     * @param width                     step width for slice expansion
     * @param minibatchSize             minibatch size
     * @param approxThreshold           threshold for approximation used in {@link MinibatchSliceSampler#isGreaterThanSliceHeight};
     *                                  approximation is exact when this threshold is zero
     */
    public MinibatchSliceSampler(final RandomGenerator rng,
                                 final List<DATA> data,
                                 final Function<Double, Double> logPrior,
                                 final BiFunction<DATA, Double, Double> logLikelihood,
                                 final double xMin,
                                 final double xMax,
                                 final double width,
                                 final int minibatchSize,
                                 final double approxThreshold) {
        super(rng, xMin, xMax, width);
        Utils.nonEmpty(data);
        Utils.nonNull(logPrior);
        Utils.nonNull(logLikelihood);
        Utils.validateArg(minibatchSize > 1, "Minibatch size must be greater than 1.");
        ParamUtils.isPositiveOrZero(approxThreshold, "Minibatch approximation threshold must be non-negative.");
        this.data = Collections.unmodifiableList(new ArrayList<>(data));
        this.logPrior = logPrior;
        this.logLikelihood = logLikelihood;
        this.minibatchSize = minibatchSize;
        this.approxThreshold = approxThreshold;
        numDataPoints = data.size();
    }

    /**
     * Creates a new sampler for an unbounded univariate random variable, given a random number generator, a list of data,
     * a continuous, univariate, unimodal, unnormalized log probability density function
     * (assumed to be a posterior and specified by a prior and a likelihood),
     * a step width, a minibatch size, and a minibatch approximation threshold.
     * @param rng                       random number generator
     * @param data                      list of data
     * @param logPrior                  log prior component of continuous, univariate, unimodal log posterior (up to additive constant)
     * @param logLikelihood             log likelihood component of continuous, univariate, unimodal log posterior (up to additive constant)
     * @param width                     step width for slice expansion
     * @param minibatchSize             minibatch size
     * @param approxThreshold           threshold for approximation used in {@link MinibatchSliceSampler#isGreaterThanSliceHeight};
     *                                  approximation is exact when this threshold is zero
     */
    public MinibatchSliceSampler(final RandomGenerator rng,
                                 final List<DATA> data,
                                 final Function<Double, Double> logPrior,
                                 final BiFunction<DATA, Double, Double> logLikelihood,
                                 final double width,
                                 final int minibatchSize,
                                 final double approxThreshold) {
        this(rng, data, logPrior, logLikelihood, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, width, minibatchSize, approxThreshold);
    }

    /**
     * Implements the OnSlice procedure from http://proceedings.mlr.press/v33/dubois14.pdf.
     */
    @Override
    boolean isGreaterThanSliceHeight(final double xProposed,
                                     final double xSample,
                                     final double z) {
        if (xProposed < xMin || xMax < xProposed) {
            return false;
        }

        //we cache values calculated from xSample, since this method is called multiple times for the same value of xSample
        //when expanding slice interval and proposing samples
        if (xSampleCache == null || xSampleCache != xSample) {
            xSampleCache = xSample;
            logPriorCache = logPrior.apply(xSample);
            logLikelihoodsCache = new HashMap<>(numDataPoints);
        }
        if (!((xSampleCache == null && logPriorCache == null && logLikelihoodsCache == null) ||
                (xSampleCache != null && logPriorCache != null && logLikelihoodsCache != null))) {
            throw new GATKException.ShouldNeverReachHereException("Cache for xSample is in an invalid state.");
        }

        final double mu0 = (logPriorCache - logPrior.apply(xProposed) - z) / numDataPoints;
        int numDataIndicesSeen = 0;
        double logLikelihoodDifferencesMean = 0.;
        double logLikelihoodDifferencesSquaredMean = 0.;

        final int numMinibatches = Math.max(numDataPoints / minibatchSize, 1);
        final Iterator<DATA> shuffledDataIterator = numMinibatches > 1
                ? lazyShuffleIterator(rng, data)
                : data.iterator();
        for (int minibatchIndex = 0; minibatchIndex < numMinibatches; minibatchIndex++) {
            final int dataIndexStart = minibatchIndex * minibatchSize;
            final int dataIndexEnd = Math.min((minibatchIndex + 1) * minibatchSize, numDataPoints);
            final int actualMinibatchSize = dataIndexEnd - dataIndexStart;  //equals minibatchSize except perhaps for last minibatch
            final List<DATA> dataMinibatch = IntStream.range(0, actualMinibatchSize).boxed()
                    .map(i -> shuffledDataIterator.next())
                    .collect(Collectors.toList());

            double logLikelihoodDifferencesMinibatchSum = 0.;
            double logLikelihoodDifferencesSquaredMinibatchSum = 0.;
            for (final DATA dataPoint : dataMinibatch) {
//                final double logLikelihoodxSample = logLikelihoodsCache.computeIfAbsent(
//                        dataIndex, i -> logLikelihood.apply(data.get(dataIndex), xSample));
                final double logLikelihoodxSample = logLikelihood.apply(dataPoint, xSample);
                final double logLikelihoodxProposed = logLikelihood.apply(dataPoint, xProposed);
                final double logLikelihoodDifference = logLikelihoodxProposed - logLikelihoodxSample;
                logLikelihoodDifferencesMinibatchSum += logLikelihoodDifference;
                logLikelihoodDifferencesSquaredMinibatchSum += logLikelihoodDifference * logLikelihoodDifference;
            }

//            final List<Double> logLikelihoodDifferencesMinibatch = dataIndicesMinibatch.stream()
//                    .map(j -> logLikelihood.apply(data.get(j), xProposed)
//                            - logLikelihoodsCache.computeIfAbsent(j, k -> logLikelihood.apply(data.get(k), xSample)))
//                    .collect(Collectors.toList());
//            final double logLikelihoodDifferencesMinibatchSum = logLikelihoodDifferencesMinibatch.stream().mapToDouble(Double::doubleValue).sum();
//            final double logLikelihoodDifferencesSquaredMinibatchSum = logLikelihoodDifferencesMinibatch.stream().mapToDouble(x -> x * x).sum();

            logLikelihoodDifferencesMean =
                    (numDataIndicesSeen * logLikelihoodDifferencesMean + logLikelihoodDifferencesMinibatchSum) /
                            (numDataIndicesSeen + actualMinibatchSize);
            logLikelihoodDifferencesSquaredMean =
                    (numDataIndicesSeen * logLikelihoodDifferencesSquaredMean + logLikelihoodDifferencesSquaredMinibatchSum) /
                            (numDataIndicesSeen + actualMinibatchSize);
            numDataIndicesSeen += actualMinibatchSize;

            if (numDataIndicesSeen == 1 || numMinibatches == 1) {
                break;
            }

            final double s = Math.sqrt(1. - (double) numDataIndicesSeen / numDataPoints) *
                    Math.sqrt((logLikelihoodDifferencesSquaredMean - Math.pow(logLikelihoodDifferencesMean, 2)) / (numDataIndicesSeen - 1));
            final double delta = 1. - new TDistribution(null, numDataIndicesSeen - 1)
                    .cumulativeProbability(Math.abs((logLikelihoodDifferencesMean - mu0) / s));

            if (delta < approxThreshold) {
//                System.out.println(String.format("%d / %d minibatches (%d / %d data points) used.", minibatchIndex + 1, numMinibatches, numDataIndicesSeen, numDataPoints));
                break;
            }
        }
        return logLikelihoodDifferencesMean > mu0;
    }

    /**
     * To efficiently sample without replacement with the possibility of early stopping when creating minibatches,
     * we lazily shuffle to avoid unnecessarily shuffling all data.
     */
    private static <T> Iterator<T> lazyShuffleIterator(final RandomGenerator rng,
                                                       final List<T> data) {
        final int numDataPoints = data.size();

        //find first prime greater than or equal to numDataPoints
        final int nextPrime = Primes.nextPrime(numDataPoints);

        return new Iterator<T>() {
            int numSeen = 0;
            int index = rng.nextInt(numDataPoints) + 1;
            final int increment = index;

            public boolean hasNext() {
                return numSeen < data.size();
            }

            @Override
            public T next() {
                while (true) {
                    index = (index + increment) % nextPrime;
                    if (index < numDataPoints) {
                        numSeen++;
                        return data.get(index);
                    }
                }
            }
        };
    }
}