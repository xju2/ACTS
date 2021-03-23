
#include "ActsExamples/TrackFindingMLBased/TrackFindingMLBasedAlgorithm.hpp"
#include "ActsExamples/EventData/ProtoTrack.hpp"
#include "ActsExamples/EventData/Index.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"

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

  pName = PyUnicode_FromString(m_cfg.inputMLModuleName.c_str());
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


void ActsExamples::TrackFindingMLBasedAlgorithm::vector_to_pylist(
  std::vector<float> const & hids, PyObject* py_hids) {
  PyObject* pHid;
  for(int i=0; i < (int) hids.size(); ++i){
    pHid = PyFloat_FromDouble(hids.at(i));
    if (!pHid) {
      fprintf(stderr, "Cannot convert argument\n");
      break;
    }
    PyList_SetItem(py_hids, i, pHid);
  }
}


ActsExamples::ProcessCode ActsExamples::TrackFindingMLBasedAlgorithm::execute(
  const ActsExamples::AlgorithmContext& ctx) const 
{
  // Read input data
  const auto& spacepoints =
    ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Convert Input data to a list of size [num_measurements x measurement_features]
  size_t num_spacepoints = spacepoints.size();
  size_t num_features = 3; // <TODO> move this to the configuration
  std::vector<float> inputMatrix(num_spacepoints * num_features, 0.0);
  ACTS_INFO("Received " << num_spacepoints << " spacepoints");

  // <TODO> Make configurable which sp features to use.
  // vectorization is applied automatically?
  for(size_t idx=0; idx < num_spacepoints; ++idx){
    auto sp = spacepoints[idx];
    inputMatrix[num_features*idx] = sp.x();
    inputMatrix[num_features*idx+1] = sp.y();
    inputMatrix[num_features*idx+2] = sp.z();
  }

  // Convert C++ vector to Python list
  // <TODO> Check memory management
  PyObject *pArgs, *pValue;
  pArgs = PyTuple_New(1);
  pValue = PyList_New(inputMatrix.size());
  vector_to_pylist(inputMatrix, pValue);
  PyTuple_SetItem(pArgs, 0, pValue);

  // call ML-based track finding
  pValue = PyObject_CallObject(pFunc, pArgs);

  // pValue contains a list of dynamic-size lists
  // Has to perform two loops to exatract the info.
  ProtoTrackContainer protoTracks;

  if (pValue && PyList_Check(pValue)) {
    Py_ssize_t num_trks = PyList_Size(pValue);
    protoTracks.reserve(num_trks);

    PyObject *pTrk, *pSP;
    for(Py_ssize_t i=0; i < num_trks; ++i){
      pTrk = PyList_GetItem(pValue, i);
      auto protoTrack = ProtoTrack();

      if (pTrk && PyList_Check(pTrk)){
        Py_ssize_t num_sps = PyList_Size(pTrk);
        for(Py_ssize_t j=0; j < num_sps; ++j) {
          pSP = PyList_GetItem(pTrk, j);
          auto idx_sp = PyLong_AsLong(pSP);
          protoTrack.push_back(spacepoints[idx_sp].measurementIndex());
        }
      }
      protoTracks.push_back(protoTrack);
    }
  } else {
    ACTS_WARNING("ML-based Track finding failed");
    protoTracks.push_back(ProtoTrack());
  }

  // Py_Finalize(); # there are no finalize function... where should I put this?
  ACTS_INFO("Created " << protoTracks.size() << " proto tracks");
  ctx.eventStore.add(m_cfg.outputProtoTracks, std::move(protoTracks));
  return ActsExamples::ProcessCode::SUCCESS;
}