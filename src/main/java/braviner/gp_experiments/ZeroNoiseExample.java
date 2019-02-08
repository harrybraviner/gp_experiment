package braviner.gp_experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

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
    static void runExperiment(double[] observationPoints, Function<Double, Double> underlyingFunction,
                              BiFunction<Double, Double, Double> kernel, double[] testPoints) throws IOException {

        double[] observationY = VectorUtils.applyFunction(underlyingFunction, observationPoints);

        double[] trueValuesAtTestPoints = VectorUtils.applyFunction(underlyingFunction, testPoints);

        // Model as a Gaussian process to get posterior for the mean and variance at the test points.
        GPUtils.PosteriorResults results = GPUtils.getPredictionsAndStddevs(observationPoints, observationY,
                kernel, testPoints);

        // Write results to a file
        try (BufferedWriter samplePointsWriter = new BufferedWriter(new FileWriter("samplePoints.csv"))) {
            samplePointsWriter.write("x,y\n");
            for (int i=0; i < observationPoints.length; i++) {
                samplePointsWriter.write(observationPoints[i] + "," + observationY[i] + "\n");
            }
        }

        try (BufferedWriter predictionsWriter = new BufferedWriter(new FileWriter("predictions.csv"))) {
            predictionsWriter.write("x,postMean,postVar,trueY\n");
            for (int i=0; i < testPoints.length; i++) {
                predictionsWriter.write(testPoints[i] + "," + results.getMeans()[i] + "," + results.getVariances()[i] +
                        "," + trueValuesAtTestPoints[i] + "\n");
            }
        }
    }

    public static void main(String[] args) throws IOException {
        Random rng = new Random(1234);

        double x_min = 0.0;
        double x_max = 3.0;

        int N_observed = 10;
        double[] samplePoints = VectorUtils.generateRandomlySampledPoints(x_min, x_max, N_observed, rng);

        // Generate an even grid of many points that we'll use to plot the function.
        int N_star = 100;
        double[] evenlySpacedPoints = VectorUtils.generateEvenlySpacedPoints(x_min, x_max, N_star);

        runExperiment(samplePoints, ZeroNoiseExample::y, ZeroNoiseExample::kernelFunction, evenlySpacedPoints);

    }

}
