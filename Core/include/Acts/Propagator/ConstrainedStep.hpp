// This file is part of the Acts project.
//
// Copyright (C) 2018-2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Common.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <limits>
#include <sstream>

namespace Acts {

/// A constrained step class for the steppers
struct ConstrainedStep {
  using Scalar = ActsScalar;

  /// the types of constraints
  /// from accuracy - this can vary up and down given a good step estimator
  /// from actor    - this would be a typical navigation step
  /// from aborter  - this would be a target condition
  /// from user     - this is user given for what reason ever
  enum Type : int { accuracy = 0, actor = 1, aborter = 2, user = 3 };

  /// the step size tuple
  std::array<Scalar, 4> values = {
      {std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max(),
       std::numeric_limits<Scalar>::max(), std::numeric_limits<Scalar>::max()}};

  /// The Navigation direction
  NavigationDirection direction = NavigationDirection::Forward;

  /// Update the step size of a certain type
  ///
  /// Only navigation and target abortion step size
  /// updates may change the sign due to overstepping
  ///
  /// @param value is the new value to be updated
  /// @param type is the constraint type
  /// @param releaseStep Allow step size to increase again
  void update(const Scalar& value, Type type, bool releaseStep = false) {
    if (releaseStep) {
      release(type);
    }
    // The check the current value and set it if appropriate
    Scalar cValue = values[type];
    values[type] = std::abs(cValue) < std::abs(value) ? cValue : value;
  }

  /// release a certain constraint value
  /// to the (signed) biggest value available, hence
  /// it depends on the direction
  ///
  /// @param type is the constraint type to be released
  void release(Type type) {
    Scalar mvalue = (direction == NavigationDirection::Forward)
                        ? (*std::max_element(values.begin(), values.end()))
                        : (*std::min_element(values.begin(), values.end()));
    values[type] = mvalue;
  }

  /// constructor from double
  /// @param value is the user given initial value
  ConstrainedStep(Scalar value)
      : direction(value > 0. ? NavigationDirection::Forward
                             : NavigationDirection::Backward) {
    values[accuracy] *= direction;
    values[actor] *= direction;
    values[aborter] *= direction;
    values[user] = value;
  }

  /// The assignment operator from one double
  /// @note this will set only the accuracy, as this is the most
  /// exposed to the Propagator, this adapts also the direction
  ///
  /// @param value is the new accuracy value
  ConstrainedStep& operator=(const Scalar& value) {
    /// set the accuracy value
    values[accuracy] = value;
    // set/update the direction
    direction = value > 0. ? NavigationDirection::Forward
                           : NavigationDirection::Backward;
    return (*this);
  }

  /// Cast operator to double, returning the min/max value
  /// depending on the direction
  operator Scalar() const {
    if (direction == NavigationDirection::Forward) {
      return (*std::min_element(values.begin(), values.end()));
    }
    return (*std::max_element(values.begin(), values.end()));
  }

  /// Access to a specific value
  ///
  /// @param type is the resquested parameter type
  Scalar value(Type type) const { return values[type]; }

  /// Return the maximum step constraint
  /// @return The max step constraint
  Scalar max() const {
    return (*std::max_element(values.begin(), values.end()));
  }

  /// Return the minimum step constraint
  /// @return The min step constraint
  Scalar min() const {
    return (*std::min_element(values.begin(), values.end()));
  }

  /// Access to currently leading min type
  ///
  Type currentType() const {
    if (direction == NavigationDirection::Forward) {
      return Type(std::min_element(values.begin(), values.end()) -
                  values.begin());
    }
    return Type(std::max_element(values.begin(), values.end()) -
                values.begin());
  }

  /// return the split value as string for debugging
  std::string toString() const;

  /// Number of iterations needed by the stepsize finder
  /// (e.g. Runge-Kutta) of the stepper.
  size_t nStepTrials = std::numeric_limits<size_t>::max();
};

inline std::string ConstrainedStep::toString() const {
  std::stringstream dstream;

  // Helper method to avoid unreadable screen output
  auto streamValue = [&](ConstrainedStep cstep) -> void {
    Scalar val = values[cstep];
    dstream << std::setw(5);
    if (std::abs(val) == std::numeric_limits<Scalar>::max()) {
      dstream << (val > 0 ? "+∞" : "-∞");
    } else {
      dstream << val;
    }
  };

  dstream << "(";
  streamValue(accuracy);
  dstream << ", ";
  streamValue(actor);
  dstream << ", ";
  streamValue(aborter);
  dstream << ", ";
  streamValue(user);
  dstream << " )";
  return dstream.str();
}

}  // namespace Acts
