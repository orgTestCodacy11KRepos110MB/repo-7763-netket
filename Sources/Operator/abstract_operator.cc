// Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Operator/abstract_operator.hpp"

#include "Machine/abstract_machine.hpp"

namespace netket {

void AbstractOperator::FindConn(VectorConstRefType v,
                                Eigen::SparseMatrix<double>& delta_v,
                                Eigen::VectorXcd& mels) const {
  delta_v.setZero();

  std::vector<Complex> weights;
  std::vector<std::vector<int>> connectors;
  std::vector<std::vector<double>> newconfs;

  FindConn(v, weights, connectors, newconfs);

  delta_v.resize(connectors.size(), v.size());

  mels.resize(connectors.size());

  for (size_t k = 0; k < connectors.size(); k++) {
    for (size_t c = 0; c < connectors[k].size(); c++) {
      delta_v.insert(k, connectors[k][c]) =
          newconfs[k][c] - v[connectors[k][c]];
    }
    mels(k) = weights[k];
  }
}

void AbstractOperator::ForEachConn(VectorConstRefType v,
                                   ConnCallback callback) const {
  std::vector<Complex> weights;
  std::vector<std::vector<int>> connectors;
  std::vector<std::vector<double>> newconfs;

  FindConn(v, weights, connectors, newconfs);

  for (size_t k = 0; k < connectors.size(); k++) {
    const ConnectorRef conn{weights[k], connectors[k], newconfs[k]};
    callback(conn);
  }
}

Eigen::VectorXcd LocalValues(Eigen::Ref<const RowMatrix<double>> samples,
                             AbstractMachine& machine,
                             const AbstractOperator& op, Index batch_size) {
  if (batch_size < 1) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
  Eigen::VectorXcd locals(samples.rows());

  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;
  Eigen::VectorXcd outlvd;

  for (auto i = Index{0}; i < samples.rows(); ++i) {
    auto v = Eigen::Ref<const Eigen::VectorXd>{samples.row(i)};

    op.FindConn(v, mel, tochange, newconfs);
    outlvd.resize(newconfs.size());
    machine.LogValDiff(v, tochange, newconfs, outlvd);

    Eigen::Map<const Eigen::ArrayXcd> meleig(&mel[0], mel.size());
    locals(i) = (meleig * outlvd.array().exp()).sum();
  }
  assert(samples.rows() > 0);

  return locals;
}

Eigen::VectorXcd LocalValuesOpOp(Eigen::Ref<const RowMatrix<double>> samples,
                                 AbstractDensityMatrix& machine,
                                 const AbstractOperator& op, Index batch_size) {
  if (batch_size < 1) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
  Eigen::VectorXcd locals(samples.rows());

  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;
  Eigen::VectorXcd outlvd;

  for (auto i = Index{0}; i < samples.rows(); ++i) {
    auto v = Eigen::Ref<const Eigen::VectorXd>{samples.row(i)};

    op.FindConn(v, mel, tochange, newconfs);
    outlvd.resize(newconfs.size());
    machine.LogValDiffRow(v, v, tochange, newconfs, outlvd);

    Eigen::Map<const Eigen::ArrayXcd> meleig(&mel[0], mel.size());
    locals(i) = (meleig * outlvd.array().exp()).sum();
  }
  assert(samples.rows() > 0);

  return locals;
}

RowMatrix<Complex> DerLocalValues(Eigen::Ref<const RowMatrix<double>> samples,
                                  AbstractMachine& machine,
                                  const AbstractOperator& op, Index batch_size,
                                  bool subtract_v_derivative) {
  if (batch_size < 1) {
    std::ostringstream msg;
    msg << "invalid batch size: " << batch_size << "; expected >=1";
    throw InvalidInputError{msg.str()};
  }
  RowMatrix<Complex> der_local(samples.rows(), machine.Npar());
  Eigen::VectorXcd val_local(samples.rows());

  std::vector<Complex> mel;
  std::vector<std::vector<int>> tochange;
  std::vector<std::vector<double>> newconfs;
  Eigen::VectorXcd outlvd;
  Eigen::Map<const Eigen::ArrayXcd> meleig(&mel[0], mel.size());

  std::vector<Complex> loc_storage;
  Eigen::Map<Eigen::ArrayXcd> loc(&loc_storage[0], loc_storage.size());

  RowMatrix<Complex> der_log_diff(1, machine.Npar());

  for (auto i = Index{0}; i < samples.rows(); ++i) {
    auto v = Eigen::Ref<const Eigen::VectorXd>{samples.row(i)};

    op.FindConn(v, mel, tochange, newconfs);
    outlvd.resize(newconfs.size());
    der_log_diff.resize(newconfs.size(), machine.Npar());

    machine.LogValDiff(v, tochange, newconfs, outlvd);
    if (subtract_v_derivative) {
      machine.DerLogDiff(v, tochange, newconfs, der_log_diff);
    } else {
      machine.DerLogChanged(v, tochange, newconfs, der_log_diff);
    }

    new (&meleig) Eigen::Map<const Eigen::ArrayXcd>(&mel[0], mel.size());

    // If we need more memory than what is allocated...
    if (meleig.size() > loc_storage.size()) {
      loc_storage.resize(meleig.size());
    }
    new (&loc) Eigen::Map<Eigen::ArrayXcd>(&loc_storage[0], meleig.size());

    loc = meleig * outlvd.array().exp();

    der_local.row(i) =
        (der_log_diff.array().colwise() * loc.array()).colwise().sum();
  }
  return der_local;
}

}  // namespace netket
