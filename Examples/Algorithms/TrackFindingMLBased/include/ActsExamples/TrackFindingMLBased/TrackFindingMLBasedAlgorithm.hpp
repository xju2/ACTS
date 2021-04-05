#pragma once

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ActsExamples/EventData/SimSpacePoint.hpp"

#include "ActsExamples/Framework/BareAlgorithm.hpp"

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

    virtual ~TrackFindingMLBasedAlgorithm() {
      if (Py_IsInitialized()) Py_Finalize();
    }

    /// Framework execute method of the track finding algorithm
    ///
    /// @param ctx is the algorithm context that holds event-wise information
    /// @return a process code to steer the algorithm flow
    ActsExamples::ProcessCode execute(
        const ActsExamples::AlgorithmContext& ctx) const final;
    ActsExamples::ProcessCode execute() const;        

    private:

      bool init_python();
      static void vector_to_pylist(std::vector<float> const & hids, PyObject* py_hids);
      // configuration
      Config m_cfg;

      // pointer to the python function
      PyObject *_pFunc;
};

}