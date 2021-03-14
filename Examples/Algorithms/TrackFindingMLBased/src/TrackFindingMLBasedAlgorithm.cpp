
#include "ActsExamples/TrackFindingMLBased/TrackFindingMLBasedAlgorithm.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"

#include <vector>
#include <stdexcept>

// Why IAlg does not have "finalize?"
// <TODO> automate following biolplate constructor.
ActsExamples::TrackFindingMLBasedAlgorithm::TrackFindingMLBasedAlgorithm(
    Config cfg, Acts::Logging::Level level)
    : ActsExamples::BareAlgorithm("TrackFindingMLBasedAlgorithm", level),
      m_cfg(std::move(cfg)) {
  if (m_cfg.inputSpacePoints.empty()) {
    throw std::invalid_argument("Missing spacepoint input collection");
  }
  if (m_cfg.outputProtoTracks.empty()) {
    throw std::invalid_argument("Missing protoTrack output collection");
  }

  if (m_cfg.inputMLModuleName.empty()) {
    throw std::invalid_argument("Missing python module name");
  }
  if (m_cfg.inputFuncName.empty()) {
    throw std::invalid_argument("Missing python function name");
  }

  init_python();
}

bool ActsExamples::TrackFindingMLBasedAlgorithm::init_python(){
  pFunc = nullptr;
  PyObject *pName, *pModule;
  // Initialize Python session
  Py_Initialize();

  pName = PyUnicode_FromString(m_cfg.inputMLModuleName);
  if (pName == NULL){
    PyErr_Print();
    throw std::runtime_error("String to unicode conversion failed");
  }
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if(pModule != NULL) {
    // Import python function
    pFunc = PyObject_GetAttrString(pModule, m_cfg.inputFuncName.c_str());
    Py_DECREF(pModule);
    if (pFunc && PyCallable_Check(pFunc)) {
      return true;
    }
  } else {
      PyErr_Print();
      throw std::runtime_error("Failed to load track finding function");
  }
  return false;
}

ActsExamples::ProcessCode ActsExamples::TrackFindingMLBasedAlgorithm::execute(
  const ActsExamples::AlgorithmContext& ctx) const 
{
  // Read input data
  const auto& spacepoints =
    ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Convert Input data to a list of size [num_measurements x measurement_features]
  std::vector<float> inputMatrix;
  size_t num_spacepoints = sourceLinks.size();
  size_t num_features = 3; // <TODO> move this to the configuration
  inputMatrix.reserve(num_spacepoints * num_features);


  // <TODO> Make configurable which sp features to use.
  for(size_t idx=0; idx < num_spacepoints; ++idx){
    auto sp = spacepoints[idx];
    inputMatrix[num_features*idx] = sp.x();
    inputMatrix[num_features*idx+1] = sp.y();
    inputMatrix[num_features*idx+2] = sp.z();
  }

  PyObject* pArgs;
  
  // Prepare the outgoing proto tracks (list of lists)
  ProtoTrackContainer protoTracks;
  

}