// This file is part of the Acts project.
//
// Copyright (C) 2021 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Definitions/TrackParametrization.hpp"
#include "Acts/EventData/Measurement.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/GeometryHierarchyMap.hpp"
#include "Acts/Geometry/GeometryIdentifier.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/Digitization/DigitizationConfig.hpp"
#include "ActsExamples/Digitization/SmearingAlgorithm.hpp"
#include "ActsExamples/EventData/Cluster.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/Measurement.hpp"
#include "ActsExamples/EventData/SimHit.hpp"
#include "ActsExamples/Framework/WriterT.hpp"

#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include <string>

namespace ActsExamples {

/// @class CsvSpacePointWriter
///
/// This writes one file per event containing information about the
/// spacepoints
///
///     event000000001-sp.csv
///     event000000002-sp.csv
///     ...
///
/// Intrinsically thread-safe as one file per event.
class CsvSpacepointsWriter final : public WriterT<SimSpacePointContainer> {
 public:
  struct Config {
    /// Which measurement collection to write.
    std::string inputSpacepoints;
    // /// Which cluster collection to write (optional)
    // std::string inputClusters;
    // /// Which simulated (truth) hits collection to use.
    // std::string inputSimHits;
    // /// Input collection to map measured hits to simulated hits.
    // std::string inputMeasurementSimHitsMap;
    /// Where to place output files
    std::string outputDir;
    /// Number of decimal digits for floating point precision in output.
    size_t outputPrecision = std::numeric_limits<float>::max_digits10;
  };

  /// Constructor with
  /// @param cfg configuration struct
  /// @param output logging level
  CsvSpacepointsWriter(const Config& cfg, Acts::Logging::Level lvl);

  /// Virtual destructor
  ~CsvSpacepointsWriter() final override;

  /// End-of-run hook
  ProcessCode endRun() final override;

 protected:
  /// This implementation holds the actual writing method
  /// and is called by the WriterT<>::write interface
  ///
  /// @param ctx The Algorithm context with per event information
  /// @param spacepoints is the data to be written out
  ProcessCode writeT(const AlgorithmContext& ctx,
                     const SimSpacePointContainer& spacepoints) final override;

 private:
  Config m_cfg;
};

}  // namespace ActsExamples
