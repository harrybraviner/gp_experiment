package braviner.gp_experiments;

import java.util.Random;
import com.github.fommil.netlib.LAPACK;
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

    public static void main(String[] args) {

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
            // Only need to populate the upper-triangular part
            for (int j=i; j < N_observed; j++) {
                matrix_K[i*N_observed + j] = kernelFunction(samplePoints[i], samplePoints[j]);
            }
        }

        double[] matrix_K_inverse = matrix_K.clone();

        for (int i=0; i < N_observed; i++) {
            System.out.print("[");
            for (int j=0; j < N_observed; j++) {
                System.out.print(matrix_K_inverse[i*N_observed + j] + "\t");
            }
            System.out.println("]");
        }
        System.out.println();
        System.out.println();

        double[] work = new double[N_observed*N_observed];
        int[] ipiv = new int[N_observed];
        intW info = new intW(0);
        lapack.dsytrf("U", N_observed, matrix_K_inverse, N_observed, ipiv, work, N_observed*N_observed, info);

        System.out.println("Info: " + info.val);
        System.out.println();

        System.out.print("[");
        for (int i=0; i < ipiv.length; i++) {
            System.out.print(ipiv[i] + "\t");
        }
        System.out.println("]");
        System.out.println();

        for (int i=0; i < N_observed; i++) {
            System.out.print("[");
            for (int j=0; j < N_observed; j++) {
                System.out.print(matrix_K_inverse[i*N_observed + j] + "\t");
            }
            System.out.println("]");
        }
        System.out.println();
        System.out.println();

    }

}
