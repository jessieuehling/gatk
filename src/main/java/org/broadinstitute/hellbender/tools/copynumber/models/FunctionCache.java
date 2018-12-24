package org.broadinstitute.hellbender.tools.copynumber.models;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * @author Samuel Lee &lt;slee@broadinstitute.org&gt;
 */
final class FunctionCache extends LinkedHashMap<Double, Double> {
    private static final long serialVersionUID = 19841647L;
    private static final int MAX_SIZE = 100_000;

    private final Function<Double, Double> mappingFunction;

    FunctionCache(final Function<Double, Double> mappingFunction) {
        this.mappingFunction = mappingFunction;
    }

    Double computeIfAbsent(final Double key) {
        return super.computeIfAbsent(key, mappingFunction);
    }

    @Override
    protected boolean removeEldestEntry(final Map.Entry<Double, Double> eldest) {
        return size() >= MAX_SIZE;
    }
}
