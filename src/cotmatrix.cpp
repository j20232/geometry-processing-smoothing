#include "cotmatrix.h"
#include <igl/doublearea.h>

double cot3(double l1, double l2, double l3, double area) {
  double c3 = (l1 * l1 + l2 * l2 - l3 * l3) / (4.0 * area);
  return c3;
}

void cotmatrix(const Eigen::MatrixXd& l, const Eigen::MatrixXi& F,
               Eigen::SparseMatrix<double>& L) {
  int num_v = F.maxCoeff() + 1;
  L.resize(num_v, num_v);  // L: V x V

  typedef Eigen::Triplet<double> T;
  std::vector<T> triplet_list;

  Eigen::VectorXd areas(F.rows());
  igl::doublearea(l, 0., areas);
  areas = areas / 2.0;

  for (int i = 0; i < F.rows(); i++) {
    for (int j = 0; j < 3; j++) {
      double cot =
          cot3(l(i, (j + 1) % 3), l(i, (j + 2) % 3), l(i, j), areas(i));
      cot = cot / 2.0;

      int idx1 = F(i, (j + 1) % 3);
      int idx2 = F(i, (j + 2) % 3);

      triplet_list.push_back(T(idx1, idx2, cot));
      triplet_list.push_back(T(idx2, idx1, cot));

      triplet_list.push_back(T(idx1, idx1, -cot));
      triplet_list.push_back(T(idx2, idx2, -cot));
    }
  }

  L.setFromTriplets(triplet_list.begin(), triplet_list.end());
}
