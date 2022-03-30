// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/TrackFindingExaTrkX/TrackFindingAlgorithmExaTrkX.hpp"

#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/SimSpacePoint.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

#include <cmath>

ActsExamples::TrackFindingAlgorithmExaTrkX::TrackFindingAlgorithmExaTrkX(
    Config config, Acts::Logging::Level level)
    : ActsExamples::BareAlgorithm("TrackFindingMLBasedAlgorithm", level),
      m_cfg(std::move(config)) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing spacepoint input collection");
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing protoTrack output collection");
  }
  if (!m_cfg.trackFinderML) {
    throw std::invalid_argument("Missing track finder");
  }
}

ActsExamples::ProcessCode ActsExamples::TrackFindingAlgorithmExaTrkX::execute(
    const ActsExamples::AlgorithmContext& ctx) const {
  // Read input data
  const auto& spacepoints =
      ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Convert Input data to a list of size [num_measurements x
  // measurement_features]
  size_t num_spacepoints = spacepoints.size();
  ACTS_INFO("Received " << num_spacepoints << " spacepoints");

  std::vector<float> inputValues;
  std::vector<uint32_t> spacepointIDs;
  inputValues.reserve(spacepoints.size() * 3);
  spacepointIDs.reserve(spacepoints.size());
  for (const auto& sp : spacepoints) {
    float x = sp.x();
    float y = sp.y();
    float z = sp.z() / 1000.;
    float r = sp.r() / 1000.;
    float phi = std::atan2(y, x) / M_PI;
    inputValues.push_back(r);
    inputValues.push_back(phi);
    inputValues.push_back(z);

    spacepointIDs.push_back(sp.measurementIndex());
  }

  // ProtoTrackContainer protoTracks;
  std::vector<std::vector<uint32_t> > trackCandidates;
  m_cfg.trackFinderML->getTracks(inputValues, spacepointIDs, trackCandidates);

  std::vector<ProtoTrack> protoTracks;
  protoTracks.reserve(trackCandidates.size());
  for (auto& x : trackCandidates) {
    if (x.size() < 4) {
      continue;
    }
    ProtoTrack onetrack;
    std::copy(x.begin(), x.end(), std::back_inserter(onetrack));
    protoTracks.push_back(std::move(onetrack));
  }

  ACTS_INFO("Created " << protoTracks.size() << " proto tracks");
  ctx.eventStore.add(m_cfg.outputProtoTracks, std::move(protoTracks));

  return ActsExamples::ProcessCode::SUCCESS;
}
