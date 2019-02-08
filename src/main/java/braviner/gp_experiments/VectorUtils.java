package braviner.gp_experiments;

import java.util.Random;
import java.util.function.Function;

public class VectorUtils {

    /**
     *
     * @param x_min Start point (inclusive)
     * @param x_max End point (inclusive)
     * @param N Number of points to generate (>= 2)
     * @return Array of evenly spaced points.
     */
    static double[] generateEvenlySpacedPoints(double x_min, double x_max, int N) {
        double[] output = new double[N];
        for (int i=0; i < N; i++) {
            output[i] = x_min + (x_max - x_min) * i / (N - 1.0);
        }

        return output;
    }

    /**
     *
     * @param x_min Start point (inclusive)
     * @param x_max End point (exclusive)
     * @param N Number of points to generate (>= 2)
     * @return Array of uniformly distributed points.
     */
    static double[] generateRandomlySampledPoints(double x_min, double x_max, int N, Random rng) {
        double[] output = new double[N];
        for (int i=0; i < N; i++) {
            output[i] = x_min + (x_max - x_min)*rng.nextDouble();
        }

        return output;
    }

    /**
     * Apply a function to many values.
     * @param f Function to apply.
     * @param xValues Input values.
     * @return Output values.
     */
    static double[] applyFunction(Function<Double, Double> f, double[] xValues) {
        double[] output = new double[xValues.length];
        for (int i=0; i < xValues.length; i++) {
            output[i] = f.apply(xValues[i]);
        }

        return output;
    }

}
