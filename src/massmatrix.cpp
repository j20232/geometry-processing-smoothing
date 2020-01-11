#include "massmatrix.h"
#include <igl/doublearea.h>

void massmatrix(const Eigen::MatrixXd& l, const Eigen::MatrixXi& F,
                Eigen::DiagonalMatrix<double, Eigen::Dynamic>& M) {
  int num_v = F.maxCoeff() + 1;
  Eigen::VectorXd areas(F.rows()), diag(num_v);

  igl::doublearea(l, 0, areas);
  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      diag(F(i, j)) += areas(i) / 6.0;
    }
  }

  M = diag.asDiagonal();
}
