#pragma once

#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include "ActsExamples/Framework/BareAlgorithm.hpp"

#include "Python.h"
#include <string>
#include <vector>

namespace ActsExamples {

class TrackFindingMLBasedAlgorithm final : public BareAlgorithm {
  public:
    struct Config {
      /// Input spacepoints collection.
      std::string inputSpacePoints;

      /// Python module name
      std::string inputMLModuleName;
      /// Python function inside the module that performs tracking finding
      std::string inputFuncName;

      /// Output protoTracks collection.
      std::string outputProtoTracks;
    };

    /// Constructor of the track finding algorithm
    ///
    /// @param cfg is the config struct to configure the algorithm
    /// @param level is the logging level
    TrackFindingMLBasedAlgorithm(Config cfg, Acts::Logging::Level lvl);

    /// Framework execute method of the track finding algorithm
    ///
    /// @param ctx is the algorithm context that holds event-wise information
    /// @return a process code to steer the algorithm flow
    ActsExamples::ProcessCode execute(
        const ActsExamples::AlgorithmContext& ctx) const final;

    private:

      bool init_python();
      static void vector_to_pylist(std::vector<float> const & hids, PyObject* py_hids);
      // configuration
      Config m_cfg;

      // pointer to the python function
      PyObject* pFunc;
};

}