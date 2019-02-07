package braviner.gp_experiments;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import com.github.fommil.netlib.LAPACK;
import com.github.fommil.netlib.BLAS;
import javafx.geometry.Pos;
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

    static class PosteriorResults {
        private double[] means;
        private double[] variances;

        PosteriorResults(double[] means, double[] variances) {
            this.means = means;
            this.variances = variances;
        }
    }

    /**
     * Returns (by mutating arrays passed in) the posterior means and variances
     * for new x-values.
     *
     * @param kernelMatrix Matrix of the prior covariances between y-values. Column-major order.
     * @param originalXValues x-values of the original observations.
     * @param originalYValues y-values of the original observations.
     * @param newXValues x-values we want predictions for.
     * @return Posterior means and variances at each point in newXValues.
     */
    static PosteriorResults getPredictionsAndStddevs(double[] kernelMatrix, double[] originalXValues,
                                        double[] originalYValues, double[] newXValues) {

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

        // Construct the matrix of covariances of new x-value and original
        // (stored in column-major order)
        double[] matrixKStar = new double[N*N_star];
        for (int i=0; i < N; i++) {
            for (int j=0; j < N_star; j++) {
                matrixKStar[i + j*N] = kernelFunction(originalXValues[i], newXValues[j]);
            }
        }

        double[] kernelMatrixClone = kernelMatrix.clone();
        int[] ipiv = new int[N];

        // We'll package getting the posterior mean and the posterior variance into one calculation
        double[] rhs = Arrays.copyOf(matrixKStar, N*(N_star + 1));
        for (int i=0; i < N; i++) {
            rhs[N*N_star + i] = originalYValues[i];
        }

        int lWork = N;
        double[] work = new double[N];
        intW info = new intW(0);
        // This populates originalYValuesClone with the solutions to the linear equation K x = [K_*, f]
        lapack.dsysv("U", N, (N_star + 1), kernelMatrixClone, N, ipiv, rhs, N_star, work, lWork, info);

        double[] outputMemory = new double[N_star*(N_star + 1)];
        // This populates outputMemory with K_*^T acting on the K^(-1) [K_*, f]
        blas.dgemm("T", "N", N_star, (N_star + 1), N, 1.0, matrixKStar, N, rhs, N, 0.0, outputMemory, N_star);

        double[] posteriorMeans = new double[N_star];
        for (int i=0; i<N_star; i++) {
            posteriorMeans[i] = outputMemory[N_star*N_star + i];
        }

        // Construct the matrix of prior covariances of the new points,
        // and subtract K_*^T K^-1 K_* at the same time.
        double[] posteriorVariances = new double[N_star*N_star];
        for (int i=0; i < N_star; i++) {
            for (int j=0; j < N_star; j++) {
                posteriorVariances[i + j*N_star] = kernelFunction(newXValues[i], newXValues[j])
                        - outputMemory[i + j*N_star];
            }
        }

        return new PosteriorResults(posteriorMeans, posteriorVariances);

    }

    public static void main(String[] args) throws IOException {

        LAPACK lapack = LAPACK.getInstance();

        Random rng = new Random(1234);

        double x_min = 0.0;
        double x_max = 3.0;

        int N_observed = 6;
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

        PosteriorResults results = getPredictionsAndStddevs(matrix_K, samplePoints, sampleY, evenlySpacedPoints);

        try (BufferedWriter samplePointsWriter = new BufferedWriter(new FileWriter("samplePoints.csv"))) {
            samplePointsWriter.write("x,y\n");
            for (int i=0; i < samplePoints.length; i++) {
                samplePointsWriter.write(samplePoints[i] + "," + sampleY[i] + "\n");
            }
        }

        try (BufferedWriter predictionsWriter = new BufferedWriter(new FileWriter("predictions.csv"))) {
            predictionsWriter.write("x,postMean,postVar\n");
            for (int i=0; i < evenlySpacedPoints.length; i++) {
                predictionsWriter.write(evenlySpacedPoints[i] + "," + results.means[i] + "," + results.variances[i] + "\n");
            }
        }

    }

}
