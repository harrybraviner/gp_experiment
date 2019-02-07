package braviner.gp_experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import com.github.fommil.netlib.LAPACK;
import com.github.fommil.netlib.BLAS;
import org.netlib.util.intW;

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
//        throw new RuntimeException("Not implemented");
    }

    /**
     * Returns (by mutating arrays passed in) the posterior means and variances
     * for new x-values.
     *
     * @param kernelMatrix Matrix of the prior covariances between y-values. Column-major order.
     * @param originalXValues x-values of the original observations.
     * @param originalYValues y-values of the original observations.
     * @param newXValues x-values we want predictions for.
     * @param outMeans Posterior means at the new x-values.
     * @param outStddevs Posterior standard deviations at the new x-values.
     */
    static void getPredictionsAndStddevs(double[] kernelMatrix, double[] originalXValues,
                                         double[] originalYValues, double[] newXValues,
                                         double[] outMeans, double[] outStddevs) {

        LAPACK lapack = LAPACK.getInstance();
        BLAS blas = BLAS.getInstance();

        // Pull out N and N* and check that dimensions all make sense.
        int N = (int)Math.sqrt(kernelMatrix.length);
        if (N*N != kernelMatrix.length) {
            throw new IllegalArgumentException("kernelMatrix was not square.");
        }
        if (originalXValues.length != N) {
            throw new IllegalArgumentException("originalXValues size does not match kernelMatrix.");
        }
        if (originalYValues.length != N) {
            throw new IllegalArgumentException("originalYValues size does not match kernelMatrix.");
        }

        int N_star = newXValues.length;
        if (outMeans.length != N_star) {
            throw new IllegalArgumentException("outMeans size does not match newXValues.");
        }
        if (outStddevs.length != N_star) {
            throw new IllegalArgumentException("outStddevs size does not match newXValues.");
        }

        // Construct the matrix of covariances of new x-value and original
        // (stored in column-major order)
        double[] matrixKStar = new double[N*N_star];
        for (int i=0; i < N; i++) {
            for (int j=0; j < N_star; j++) {
                matrixKStar[i + j*N] = kernelFunction(originalXValues[i], newXValues[j]);
            }
        }

        // FIXME - compute K^-1 (f - 0) using dsysv.
        // FIXME - then act with K_*^T, that will give us the mean.

        double[] kernelMatrixClone = kernelMatrix.clone();
        int[] ipiv = new int[N];
        double[] originalYValuesClone = originalYValues.clone();
        int lWork = N;
        double[] work = new double[N];
        intW info = new intW(0);
        // This populates originalYValuesClone with the solutions to the linear equation K x = f
        lapack.dsysv("U", N, N, kernelMatrixClone, N, ipiv, originalYValuesClone, N_star, work, lWork, info);

        // This populate muStarOutput with K_*^T acting on the K^(-1) f
        blas.dgemv("T", N, N_star, 1.0, matrixKStar, N, originalYValuesClone, 1, 0.0, outMeans, 1);

    }

    public static void main(String[] args) throws IOException {

        LAPACK lapack = LAPACK.getInstance();

        Random rng = new Random(1234);

        double x_min = 0.0;
        double x_max = 3.0;

        int N_observed = 10;
        double[] samplePoints = new double[N_observed];
        for (int i=0; i < N_observed; i++) {
            samplePoints[i] = x_min + (x_max - x_min)*rng.nextDouble();
        }

        double[] sampleY = new double[N_observed];
        for (int i=0; i < N_observed; i++) {
            sampleY[i] = y(samplePoints[i]);
        }

        double[] matrix_K = new double[N_observed*N_observed];
        for (int i=0; i < N_observed; i++) {
            // Populate the whole thing since I'll use general matrix routines for now
            for (int j=0; j < N_observed; j++) {
                matrix_K[i + j*N_observed] = kernelFunction(samplePoints[i], samplePoints[j]);
            }
        }

        // Generate an even grid of many points that we'll use to plot the function.
        int N_star = 100;
        double[] evenlySpacedPoints = new double[N_star];
        for (int i=0; i < N_star; i++) {
            evenlySpacedPoints[i] = x_min + (x_max - x_min) * i / (N_star - 1.0);
        }

        double[] posteriorMeans = new double[N_star];
        double[] posteriorStddevs = new double[N_star];
        getPredictionsAndStddevs(matrix_K, samplePoints, sampleY, evenlySpacedPoints,
                posteriorMeans, posteriorStddevs);

        try (BufferedWriter samplePointsWriter = new BufferedWriter(new FileWriter("samplePoints.csv"))) {
            samplePointsWriter.write("x,y\n");
            for (int i=0; i < samplePoints.length; i++) {
                samplePointsWriter.write(samplePoints[i] + "," + sampleY[i] + "\n");
            }
        }

        try (BufferedWriter predictionsWriter = new BufferedWriter(new FileWriter("predictions.csv"))) {
            predictionsWriter.write("x,postMean\n");
            for (int i=0; i < evenlySpacedPoints.length; i++) {
                predictionsWriter.write(evenlySpacedPoints[i] + "," + posteriorMeans[i] + "\n");
            }
        }

    }

}
