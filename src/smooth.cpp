#include "smooth.h"
#include <igl/edge_lengths.h>
#include "cotmatrix.h"
#include "massmatrix.h"

void smooth(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
            const Eigen::MatrixXd& G, double lambda, Eigen::MatrixXd& U) {
  // Edge lengths
  Eigen::MatrixXd edge_lengths;
  igl::edge_lengths(V, F, edge_lengths);

  // Cot matrix
  Eigen::SparseMatrix<double> L;
  cotmatrix(edge_lengths, F, L);

  // Mass matrix
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> M;
  massmatrix(edge_lengths, F, M);

  Eigen::SparseMatrix<double> A = -lambda * L;
  for (int i = 0; i < A.rows(); i++) A.coeffRef(i, i) += M.diagonal()[i];

  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> cholesky;
  cholesky.compute(A);
  U = cholesky.solve(M * G);
}
