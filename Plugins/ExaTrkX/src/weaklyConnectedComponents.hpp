// This file is part of the Acts project.
//
// Copyright (C) 2022 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#ifndef ExaTrkX_USE_CUDA
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

namespace ExaTrkX {
template <typename vertex_t, typename edge_t, typename weight_t>
void weaklyConnectedComponents(
  vertex_t numNodes,
  std::vector<vertex_t>& rowIndices,
  std::vector<vertex_t>& colIndices,
  std::vector<weight_t>& edgeWeights,
  std::vector<vertex_t>& trackLabels) 
{
  typedef
    boost::adjacency_list<
    boost::vecS            // edge list
  , boost::vecS            // vertex list
  , boost::undirectedS     // directedness
  , boost::no_property     // property associated with vertices
  , float                  // property associated with edges
  > Graph; 

  Graph g(numNodes);
  for(size_t idx=0; idx < rowIndices.size(); ++idx) {
    boost::add_edge(
      rowIndices[idx], colIndices[idx], edgeWeights[idx], g);
  }
  size_t num_components = boost::connected_components(g, &trackLabels[0]);

}
}
#else

#include <boost/range/combine.hpp>
#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
  {                                                                           \
    cudaError_t cudaStatus = call;                                            \
    if (cudaSuccess != cudaStatus)                                            \
      fprintf(stderr,                                                         \
              "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
              "%s (%d).\n",                                                   \
              #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus),      \
              cudaStatus);                                                    \
  }
#endif  // CUDA_RT_CALL
namespace ExaTrkX {
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void weaklyConnectedComponents(
  vertex_t numNodes,
  std::vector<vertex_t>& rowIndices,
  std::vector<vertex_t>& colIndices,
  std::vector<weight_t>& edgeWeights,
  std::vector<vertex_t>& trackLabels) 
{
  cudaStream_t stream;
  CUDA_RT_CALL(cudaStreamCreate(&stream));

  // std::cout << "Weakly components Start" << std::endl;
  // std::cout << "edge size: " << rowIndices.size() << " " << colIndices.size()
  // << std::endl;
  raft::handle_t handle{stream};
  std::cout << "WCC with handle " << std::endl;

  cugraph::graph_t<vertex_t, edge_t, weight_t, false, false> graph(handle);

  // constexpr bool renumber = true;
  // using store_transposed = bool;

#if 0
    static int PERF = 0;
    HighResClock hr_clock{};

    if (PERF) {
      CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_clock.start();
    }
#endif

  // learn from matrix_market_file_utilities.cu
  // vertex_t maxVertexID_row =
  //     *std::max_element(rowIndices.begin(), rowIndices.end());
  // vertex_t maxVertexID_col =
  //     *std::max_element(colIndices.begin(), colIndices.end());
  // vertex_t maxVertex = std::max(maxVertexID_row, maxVertexID_col);

  vertex_t number_of_vertices = numNodes;
  rmm::device_uvector<vertex_t> d_vertices(number_of_vertices,
                                           handle.get_stream());
  std::vector<vertex_t> vertex_idx(number_of_vertices);
  for (vertex_t idx = 0; idx < number_of_vertices; idx++) {
    vertex_idx[idx] = idx;
  }

  rmm::device_uvector<vertex_t> src_v(rowIndices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> dst_v(colIndices.size(), handle.get_stream());
  rmm::device_uvector<weight_t> weights_v(edgeWeights.size(),
                                          handle.get_stream());

  std::cout << "WCC update device " << std::endl;
  raft::update_device(src_v.data(), rowIndices.data(), rowIndices.size(),
                      handle.get_stream());
  raft::update_device(dst_v.data(), colIndices.data(), colIndices.size(),
                      handle.get_stream());
  raft::update_device(weights_v.data(), edgeWeights.data(), edgeWeights.size(),
                      handle.get_stream());
  raft::update_device(d_vertices.data(), vertex_idx.data(), vertex_idx.size(),
                      handle.get_stream());

  std::cout << "WCC create_graph_from_edgelist" << std::endl;
  std::tie(graph, std::ignore) =
      cugraph::create_graph_from_edgelist<vertex_t, edge_t, weight_t, false,
                                          false>(
          handle, std::move(d_vertices), std::move(src_v), std::move(dst_v),
          std::move(weights_v), cugraph::graph_properties_t{true, false},
          false);

std::cout << "WCC created graph" << std::endl;
  auto graph_view = graph.view();
  std::cout << "WCC to sync " << std::endl;
  CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement

  std::cout << "WCC start " << std::endl;
  rmm::device_uvector<vertex_t> d_components(
      graph_view.get_number_of_vertices(), handle.get_stream());

  // std::cout << "2back from construct_graph" << std::endl;
  cugraph::weakly_connected_components(handle, graph_view, d_components.data());

  

  raft::update_host(trackLabels.data(), d_components.data(),
                    d_components.size(), handle.get_stream());

  std::cout << "Finished weakly connected components " << std::endl;
}
}
#endif