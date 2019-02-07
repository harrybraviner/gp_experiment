package braviner.gp_experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.function.Function;

import com.github.fommil.netlib.LAPACK;

public class ZeroNoiseExample {

    /**
     * Underlying function that points are drawn from.
     *
     * Sin function with a trend.
     *
     * @param x Input value.
     * @return Output value.
     */
    static double y(double x) {
        return 2.3*x + Math.sin(2.0*Math.PI * x);
    }

    /**
     * Kernel function for the prior covariance in y between two x values.
     * Must be symmetric in x1, x2.
     *
     * @param x1 First input.
     * @param x2 Second input.
     * @return Output value.
     */
    static double kernelFunction(double x1, double x2) {
        return Math.exp(-0.5 * (x1 - x2)*(x1 - x2));
    }

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

    public static void main(String[] args) throws IOException {

        LAPACK lapack = LAPACK.getInstance();

        Random rng = new Random(1234);

        double x_min = 0.0;
        double x_max = 3.0;

        int N_observed = 10;
        double[] samplePoints = generateRandomlySampledPoints(x_min, x_max, N_observed, rng);

        double[] sampleY = applyFunction(ZeroNoiseExample::y, samplePoints);

        // Generate an even grid of many points that we'll use to plot the function.
        int N_star = 100;
        double[] evenlySpacedPoints = generateEvenlySpacedPoints(x_min, x_max, N_star);

        double[] trueValues = applyFunction(ZeroNoiseExample::y, evenlySpacedPoints);

        // Model as a Gaussian process to get posterior for the mean and variance at the test points.
        GPUtils.PosteriorResults results = GPUtils.getPredictionsAndStddevs(samplePoints, sampleY,
                ZeroNoiseExample::kernelFunction, evenlySpacedPoints);

        // Write results to a file
        try (BufferedWriter samplePointsWriter = new BufferedWriter(new FileWriter("samplePoints.csv"))) {
            samplePointsWriter.write("x,y\n");
            for (int i=0; i < samplePoints.length; i++) {
                samplePointsWriter.write(samplePoints[i] + "," + sampleY[i] + "\n");
            }
        }

        try (BufferedWriter predictionsWriter = new BufferedWriter(new FileWriter("predictions.csv"))) {
            predictionsWriter.write("x,postMean,postVar,trueY\n");
            for (int i=0; i < evenlySpacedPoints.length; i++) {
                predictionsWriter.write(evenlySpacedPoints[i] + "," + results.getMeans()[i] + "," + results.getVariances()[i] +
                        "," + trueValues[i] + "\n");
            }
        }

    }

}
