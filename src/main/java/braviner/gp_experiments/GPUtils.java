package braviner.gp_experiments;

import com.github.fommil.netlib.BLAS;
import com.github.fommil.netlib.LAPACK;
import org.netlib.util.intW;

import java.util.Arrays;
import java.util.function.BiFunction;

public class GPUtils {


    /**
     * Wrapper class to allow me to return two arrays.
     */
    static class PosteriorResults {
        private double[] means;
        private double[] variances;

        PosteriorResults(double[] means, double[] variances) {
            this.means = means;
            this.variances = variances;
        }

        public double[] getMeans() {
            return means;
        }

        public double[] getVariances() {
            return variances;
        }
    }

    /**
     * Generate a matrix with elements K[i ,j] = kernel(x1[i], x2[j])
     * in column-major order.
     * @param x1 Input vector.
     * @param x2 Input vector.
     * @return Matrix in column-major order.
     */
    static double[] generateKernelMatrix(BiFunction<Double, Double, Double> kernelFunction,
                                         double[] x1, double[] x2, double noiseVar) {
        int N1 = x1.length;
        int N2 = x2.length;
        double[] kernelMatrix = new double[N1*N2];

        for (int i=0; i < N1; i++) {
            for (int j=0; j < N2; j++) {
                kernelMatrix[i + j*N1] = kernelFunction.apply(x1[i], x2[j]) + (i == j ? noiseVar : 0.0);
            }
        }

        return kernelMatrix;
    }

    /**
     * Returns (by mutating arrays passed in) the posterior means and variances
     * for new x-values.
     *
     * @param originalXValues x-values of the original observations.
     * @param originalYValues y-values of the original observations.
     * @param newXValues x-values we want predictions for.
     * @return Posterior means and variances at each point in newXValues.
     */
    static PosteriorResults getPredictionsAndStddevs(double[] originalXValues, double[] originalYValues,
                                                     BiFunction<Double, Double, Double> kernelFunction,
                                                     double noiseVar,
                                                     double[] newXValues) {
        double[] kernelMatrix = generateKernelMatrix(kernelFunction, originalXValues, originalXValues, noiseVar);
        return getPredictionsAndStddevs(kernelMatrix, originalXValues, originalYValues, kernelFunction, newXValues);
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
                                                     double[] originalYValues,
                                                     BiFunction<Double, Double, Double> kernelFunction,
                                                     double[] newXValues) {

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
        // N.B.: The noise variance should not appear in this matrix.
        double[] matrixKStar = new double[N*N_star];
        for (int i=0; i < N; i++) {
            for (int j=0; j < N_star; j++) {
                matrixKStar[i + j*N] = kernelFunction.apply(originalXValues[i], newXValues[j]);
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
        // N.B.: The noise variance should not appear in the K_starstar matrix, so we don't need it in here.
        double[] posteriorVariances = new double[N_star*N_star];
        for (int i=0; i < N_star; i++) {
            for (int j=0; j < N_star; j++) {
                posteriorVariances[i + j*N_star] = kernelFunction.apply(newXValues[i], newXValues[j])
                        - outputMemory[i + j*N_star];
            }
        }

        return new PosteriorResults(posteriorMeans, posteriorVariances);

    }
}
