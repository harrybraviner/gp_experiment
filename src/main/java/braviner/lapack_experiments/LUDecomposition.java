package braviner.lapack_experiments;

import com.github.fommil.netlib.LAPACK;
import org.netlib.util.intW;

public class LUDecomposition {

    /**
     * Prints a matrix stored in column-major order.
     * @param matrix Matrix in column major order.
     * @param lda Leading dimension (i.e. number of rows).
     */
    static void printMatrix(double[] matrix, int lda) {
        int N = matrix.length;
        if (N % lda != 0) {
            throw new IllegalArgumentException("lda must divide size of matrix");
        }
        int nCols = N / lda;
        for (int i=0; i < lda; i++) {
            System.out.print("[\t");
            for (int j=0; j < nCols; j++) {
                System.out.print(String.format("%6.3f\t", matrix[i + j*lda]));
            }
            System.out.println("]");
        }
    }

    /**
     * Interchanges (in-place) two rows of a matrix stored in column-major order.
     *
     * @param matrix Matrix in column major order.
     * @param lda Leading dimension (i.e. number of rows).
     * @param row1 Row to swap (0-indexed).
     * @param row2 Row to swap (0-indexed).
     */
    static void swapRows(double[] matrix, int lda, int row1, int row2) {
        int N = matrix.length;
        if (N % lda != 0) {
            throw new IllegalArgumentException("lda must divide size of matrix");
        }
        int nCols = N / lda;
        for (int i=0; i < nCols; i++) {
            double temp = matrix[row1 + i*lda];
            matrix[row1 + i*lda] = matrix[row2 + i*lda];
            matrix[row2 + i*lda] = temp;
        }
    }

    /**
     * Reverses the row permutation applied by ipiv.
     *
     * @param matrix Matrix in column major order.
     * @param lda Leading dimension (i.e. number of rows).
     * @param ipiv Permutation.
     */
    static void reversePermutation(double[] matrix, int lda, int[] ipiv) {
        int N = matrix.length;
        if (N % lda != 0) {
            throw new IllegalArgumentException("lda must divide size of matrix");
        }
        for (int i = lda-1; i >= 0; i--) {
            swapRows(matrix, lda, i, ipiv[i]-1);
        }
    }

    public static void generalSquareLU() {
        // Real matrix, no symmetry or other properties

        double[] arr1 = new double[] {
                1, 0.5, -3,
                6, 0, 1,
                4, 1, 7
        };

        int N = 3;  // Dimension

        System.out.println("A:");
        printMatrix(arr1, N);
        System.out.println();

        int[] ipiv = new int[N];

        intW info = new intW(0);

        LAPACK.getInstance().dgetrf(N, N, arr1, N, ipiv, info);

        System.out.println("Memory after call:");
        printMatrix(arr1, N);
        System.out.println();

        // Now we recover A manually to ensure that we understand
        // what the decomposition has given us.

        double[] recoveredA = new double[N*N];

        // Compute L*U. Note that L has unit diagonal elements, which are not stored.
        for (int i=0; i < N; i++) {
            for (int j=0; j < N; j++) {
                double acc = 0.0;
                for (int k = 0; k <= Math.min(i, j); k++) {
                    double u_elem = arr1[k + j*N];
                    double l_elem = k == i ? 1.0 : arr1[i + k*N];
                    acc += u_elem * l_elem;
                }
                recoveredA[i + j*N] = acc;
            }
        }

        System.out.println("Recovered A (rows permuted):");
        printMatrix(recoveredA, N);
        System.out.println();

        reversePermutation(recoveredA, N, ipiv);

        System.out.println("Recovered A (rows restored to original order):");
        printMatrix(recoveredA, N);
        System.out.println();

    }

    public static void main(String args[]) {
        generalSquareLU();

    }

}
