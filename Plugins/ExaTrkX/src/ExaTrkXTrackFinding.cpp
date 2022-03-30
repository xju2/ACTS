// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/ExaTrkX/ExaTrkXTrackFinding.hpp"

#include <counting_sort.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <find_nbrs.h>
#include <grid.h>
#include <insert_points.h>
#include <prefix_sum.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "weaklyConnectedComponents.hpp"

using namespace torch::indexing;

Acts::ExaTrkXTrackFinding::ExaTrkXTrackFinding(const Config& config)
    : m_cfg(config) {
  std::cout << "Model input directory: " << m_cfg.inputMLModuleDir << "\n";
  std::cout << "Spacepoint features: " << m_cfg.spacepointFeatures << "\n";
  std::cout << "Embedding Dimension: " << m_cfg.embeddingDim << "\n";
  std::cout << "radius value       : " << m_cfg.rVal << "\n";
  std::cout << "k-nearest neigbour : " << m_cfg.knnVal << "\n";
  std::cout << "filtering cut      : " << m_cfg.filterCut << "\n";

  std::string l_embedModelPath(m_cfg.inputMLModuleDir + "/embed.pt");
  std::string l_filterModelPath(m_cfg.inputMLModuleDir + "/filter.pt");
  std::string l_gnnModelPath(m_cfg.inputMLModuleDir + "/gnn.pt");
  c10::InferenceMode guard(true);
  try {   
      e_model = torch::jit::load(l_embedModelPath.c_str());
      e_model.eval();
      f_model = torch::jit::load(l_filterModelPath.c_str());
      f_model.eval();
      g_model = torch::jit::load(l_gnnModelPath.c_str());
      g_model.eval();
  } catch (const c10::Error& e) {
      throw std::invalid_argument("Failed to load models: " + e.msg()); 
  }
}

void Acts::ExaTrkXTrackFinding::buildEdges(std::vector<float>& embedFeatures,
                                           std::vector<int64_t>& edgeList,
                                           int64_t numSpacepoints) const {
  torch::Device device(torch::kCUDA);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  int grid_params_size;
  int grid_delta_idx;
  int grid_total_idx;
  int grid_max_res;
  int grid_dim;
  int dim = m_cfg.embeddingDim;
  if (dim >= 3) {
    grid_params_size = 8;
    grid_delta_idx = 3;
    grid_total_idx = 7;
    grid_max_res = 128;
    grid_dim = 3;
  } else {
    throw std::runtime_error("DIM < 3 is not supported for now.\n");
  }

  float cell_size;
  float radius_cell_ratio = 2.0;
  int G = -1;
  int batch_size = 1;
  float rVal = m_cfg.rVal;  // radius of nearest neighours
  int kVal = m_cfg.knnVal;  // maximum number of nearest neighbours.

  // Set up grid properties
  torch::Tensor grid_min;
  torch::Tensor grid_max;
  torch::Tensor grid_size;

  torch::Tensor embedTensor =
      torch::tensor(embedFeatures, options)
          .reshape({1, numSpacepoints, m_cfg.embeddingDim});
  torch::Tensor gridParamsCuda =
      torch::zeros({batch_size, grid_params_size}, device).to(torch::kFloat32);
  torch::Tensor r_tensor = torch::full({batch_size}, rVal, device);
  torch::Tensor lengths = torch::full({batch_size}, numSpacepoints, device);

  // build the grid
  for (int i = 0; i < batch_size; i++) {
    torch::Tensor allPoints =
        embedTensor.index({i, Slice(None, lengths.index({i}).item().to<long>()),
                           Slice(None, grid_dim)});
    grid_min = std::get<0>(allPoints.min(0));
    grid_max = std::get<0>(allPoints.max(0));
    gridParamsCuda.index_put_({i, Slice(None, grid_delta_idx)}, grid_min);

    grid_size = grid_max - grid_min;

    cell_size = r_tensor.index({i}).item().to<float>() / radius_cell_ratio;

    if (cell_size < (grid_size.min().item().to<float>() / grid_max_res)) {
      cell_size = grid_size.min().item().to<float>() / grid_max_res;
    }

    gridParamsCuda.index_put_({i, grid_delta_idx}, 1 / cell_size);

    gridParamsCuda.index_put_({i, Slice(1 + grid_delta_idx, grid_total_idx)},
                              floor(grid_size / cell_size) + 1);

    gridParamsCuda.index_put_(
        {i, grid_total_idx},
        gridParamsCuda.index({i, Slice(1 + grid_delta_idx, grid_total_idx)})
            .prod());

    if (G < gridParamsCuda.index({i, grid_total_idx}).item().to<int>()) {
      G = gridParamsCuda.index({i, grid_total_idx}).item().to<int>();
    }
  }

  torch::Tensor pc_grid_cnt =
      torch::zeros({batch_size, G}, device).to(torch::kInt32);
  torch::Tensor pc_grid_cell =
      torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);
  torch::Tensor pc_grid_idx =
      torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);

  std::cout << "Inserting points" << std::endl;

  // put spacepoints into the grid
  InsertPointsCUDA(embedTensor, lengths.to(torch::kInt64), gridParamsCuda,
                   pc_grid_cnt, pc_grid_cell, pc_grid_idx, G);

  torch::Tensor pc_grid_off =
      torch::full({batch_size, G}, 0, device).to(torch::kInt32);
  torch::Tensor grid_params = gridParamsCuda.to(torch::kCPU);

  std::cout << "Prefix Sum" << std::endl;

  for (int i = 0; i < batch_size; i++) {
    PrefixSumCUDA(pc_grid_cnt.index({i}),
                  grid_params.index({i, grid_total_idx}).item().to<int>(),
                  pc_grid_off.index({i}));
  }

  torch::Tensor sorted_points =
      torch::zeros({batch_size, numSpacepoints, dim}, device)
          .to(torch::kFloat32);
  torch::Tensor sorted_points_idxs =
      torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);

  CountingSortCUDA(embedTensor, lengths.to(torch::kInt64), pc_grid_cell,
                   pc_grid_idx, pc_grid_off, sorted_points, sorted_points_idxs);

  std::cout << "Counting sorted" << std::endl;

  // torch::Tensor K_tensor = torch::full({batch_size}, kVal, device);

  std::tuple<at::Tensor, at::Tensor> nbr_output = FindNbrsCUDA(
      sorted_points, sorted_points, lengths.to(torch::kInt64),
      lengths.to(torch::kInt64), pc_grid_off.to(torch::kInt32),
      sorted_points_idxs, sorted_points_idxs,
      gridParamsCuda.to(torch::kFloat32), kVal, r_tensor, r_tensor * r_tensor);

  std::cout << "Neigbours to Edges" << std::endl;
  torch::Tensor positiveIndices = std::get<0>(nbr_output) >= 0;

  torch::Tensor repeatRange = torch::arange(positiveIndices.size(1), device)
                                  .repeat({1, positiveIndices.size(2), 1})
                                  .transpose(1, 2);

  torch::Tensor stackedEdges =
      torch::stack({repeatRange.index({positiveIndices}),
                    std::get<0>(nbr_output).index({positiveIndices})});

  //  Remove self-loops:

  torch::Tensor selfLoopMask =
      stackedEdges.index({0}) != stackedEdges.index({1});
  stackedEdges = stackedEdges.index({Slice(), selfLoopMask});

  // Perform any other post-processing here. E.g. Can remove half of edge list
  // with:

  torch::Tensor duplicate_mask =
      stackedEdges.index({0}) > stackedEdges.index({1});
  stackedEdges = stackedEdges.index({Slice(), duplicate_mask});

  // And randomly flip direction with:
  // torch::Tensor random_cut_keep = torch::randint(2, {stackedEdges.size(1)});
  // torch::Tensor random_cut_flip = 1-random_cut_keep;
  // torch::Tensor keep_edges = stackedEdges.index({Slice(),
  // random_cut_keep.to(torch::kBool)}); torch::Tensor flip_edges =
  // stackedEdges.index({Slice(), random_cut_flip.to(torch::kBool)}).flip({0});
  // stackedEdges = torch::cat({keep_edges, flip_edges}, 1);
  stackedEdges = stackedEdges.toType(torch::kInt64).to(torch::kCPU);

  std::cout << "copy edges to std::vector" << std::endl;
  std::copy(stackedEdges.data_ptr<int64_t>(),
            stackedEdges.data_ptr<int64_t>() + stackedEdges.numel(),
            std::back_inserter(edgeList));
}

void Acts::ExaTrkXTrackFinding::getTracks(
    std::vector<float>& inputValues, std::vector<uint32_t>& spacepointIDs,
    std::vector<std::vector<uint32_t> >& trackCandidates) const {
  // hardcoded debugging information
  c10::InferenceMode guard(true);
  bool debug = true;
  const std::string embedding_outname = "debug_embedding_outputs.txt";
  const std::string edgelist_outname = "debug_edgelist_outputs.txt";
  const std::string filtering_outname = "debug_filtering_scores.txt";
  torch::Device device(torch::kCUDA);

  // printout the r,phi,z of the first spacepoint
  std::cout << "First spacepoint information: " << inputValues.size() << "\n\t";
  std::copy(inputValues.begin(), inputValues.begin() + 3,
            std::ostream_iterator<float>(std::cout, " "));
  std::cout << std::endl;

  // ************
  // Embedding
  // ************

  int64_t numSpacepoints = inputValues.size() / m_cfg.spacepointFeatures;
  std::vector<torch::jit::IValue> eInputTensorJit;
  auto e_opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor eLibInputTensor = torch::from_blob(
      inputValues.data(),
      {numSpacepoints, m_cfg.spacepointFeatures},
      e_opts).to(torch::kFloat32);
  eInputTensorJit.push_back(eLibInputTensor.to(device));
  at::Tensor eOutput = e_model.forward(eInputTensorJit).toTensor();
  std::cout <<"Embedding space of libtorch the first SP: \n";
  std::cout << eOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << std::endl;
  std::cout << std::endl;

  eOutput = eOutput.cpu();

  std::vector<float> eOutputData;
  std::copy(eOutput.data_ptr<float>(),
            eOutput.data_ptr<float>() + eOutput.numel(),
            std::back_inserter(eOutputData));

  if (debug) {
    std::fstream out(embedding_outname, out.out);
    if (!out.is_open()) {
      std::cout << "failed to open " << embedding_outname << '\n';
    } else {
      std::copy(eOutputData.begin(), eOutputData.end(),
                std::ostream_iterator<float>(out, " "));
    }
  }

  // ************
  // Building Edges
  // ************

  //TODO: use torch:tensor instead.
  std::vector<int64_t> edgeList;
  buildEdges(eOutputData, edgeList, numSpacepoints);
  int64_t numEdges = edgeList.size() / 2;
  std::cout << "Built " << numEdges << " edges." << std::endl;

  std::copy(edgeList.begin(), edgeList.begin() + 10,
            std::ostream_iterator<int64_t>(std::cout, " "));
  std::cout << std::endl;
  std::copy(edgeList.begin() + numEdges, edgeList.begin() + numEdges + 10,
            std::ostream_iterator<int64_t>(std::cout, " "));
  std::cout << std::endl;

  if (debug) {
    std::fstream out(edgelist_outname, out.out);
    if (!out.is_open()) {
      std::cout << "failed to open " << edgelist_outname << '\n';
    } else {
      std::copy(edgeList.begin(), edgeList.end(),
                std::ostream_iterator<int64_t>(out, " "));
    }
  }

  // ************
  // Filtering
  // ************
  std::cout << "Get scores for " << numEdges << " edges." << std::endl;
  // Use torch::tensor
  torch::Tensor edgeListCTen = torch::tensor(edgeList, {torch::kInt64});
  edgeListCTen = edgeListCTen.to(device);
  edgeListCTen = edgeListCTen.reshape({2, numEdges});

  std::cout << "Prepare inputs for filtering" << std::endl;

  std::vector<torch::jit::IValue> fInputTensorJit;
  fInputTensorJit.push_back(eLibInputTensor.to(device));
  fInputTensorJit.push_back(edgeListCTen.to(device));
  at::Tensor fOutputCTen = f_model.forward(fInputTensorJit).toTensor();
  fOutputCTen.squeeze_();
  fOutputCTen.sigmoid_();

  std::cout << "After filtering" << std::endl;

  // if (debug) {
  //   std::fstream out(filtering_outname, out.out);
  //   if (!out.is_open()) {
  //     std::cout << "failed to open " << filtering_outname << '\n';
  //   } else {
  //     std::copy(fOutputCTen.data_ptr<float>(),
  //               fOutputCTen.data_ptr<float>() + fOutputCTen.numel(),
  //               std::ostream_iterator<float>(out, " "));
  //   }
  // }

  // std::cout << fOutputCTen.slice(0, 0, 3) << std::endl;
  torch::Tensor filterMask = fOutputCTen > m_cfg.filterCut;
  torch::Tensor edgesAfterF = edgeListCTen.index({Slice(), filterMask});
  int64_t numEdgesAfterF = edgesAfterF.size(1);
  std::cout << "After filtering: " << numEdgesAfterF << " edges." << std::endl;

  // ************
  // GNN
  // ************
  std::vector<torch::jit::IValue> gInputTensorJit;
  // auto g_opts = torch::TensorOptions().dtype(torch::kInt64);
  gInputTensorJit.push_back(eLibInputTensor.to(device));
  gInputTensorJit.push_back(edgesAfterF.to(device));
  auto gOutputCTen = g_model.forward(gInputTensorJit).toTensor();
  gOutputCTen.sigmoid_();
  gOutputCTen = gOutputCTen.cpu();

  edgesAfterF = edgesAfterF.cpu();

  std::cout << gOutputCTen.slice(0, 0, 3) << std::endl;

  // ************
  // Track Labeling with cugraph::connected_components
  // ************
  std::vector<int32_t> rowIndices;
  std::vector<int32_t> colIndices;
  std::vector<float> edgeWeights;
  std::vector<int32_t> trackLabels(numSpacepoints);
  std::copy(edgesAfterF.data_ptr<int64_t>(),
            edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
            std::back_insert_iterator(rowIndices));
  std::copy(edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
            edgesAfterF.data_ptr<int64_t>()+edgesAfterF.numel(),
            std::back_insert_iterator(colIndices));
  std::copy(gOutputCTen.data_ptr<float>(),
            gOutputCTen.data_ptr<float>() + numEdgesAfterF,
            std::back_insert_iterator(edgeWeights));

  std::cout << "run weaklyConnectedComponents" << std::endl;
  weaklyConnectedComponents<int32_t, int32_t, float>(rowIndices, colIndices,
                                                     edgeWeights, trackLabels);

  std::cout << "size of components: " << trackLabels.size() << std::endl;
  if (trackLabels.size() == 0)
    return;

  trackCandidates.clear();

  uint32_t existTrkIdx = 0;
  // map labeling from MCC to customized track id.
  std::map<uint32_t, uint32_t> trackLableToIds;

  for (uint32_t idx = 0; idx < numSpacepoints; ++idx) {
    uint32_t trackLabel = trackLabels[idx];
    int spacepointID = spacepointIDs[idx];

    uint32_t trkId;
    if (trackLableToIds.find(trackLabel) != trackLableToIds.end()) {
      trkId = trackLableToIds[trackLabel];
      trackCandidates[trkId].push_back(spacepointID);
    } else {
      // a new track, assign the track id
      // and create a vector
      trkId = existTrkIdx;
      trackCandidates.push_back(std::vector<uint32_t>{trkId});
      trackLableToIds[trackLabel] = trkId;
      existTrkIdx++;
    }
  }
}
