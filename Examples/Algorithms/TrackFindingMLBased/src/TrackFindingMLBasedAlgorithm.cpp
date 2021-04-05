
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
  _pFunc = nullptr;
  PyObject *pName, *pModule;

  wchar_t *program = Py_DecodeLocale("/usr/bin/python", NULL);
  Py_SetProgramName(program);
  delete program;
  
  Py_InitializeEx(0);  //  it skips initialization registration of signal handlers, which might be useful when Python is embedded
  PySys_SetArgv(0, (wchar_t**)"dummy");

  pName = PyUnicode_FromString(m_cfg.inputMLModuleName.c_str());
  if (pName == NULL){
    PyErr_Print();
    throw std::runtime_error("String to unicode conversion failed");
  }
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  bool is_ready = false;
  if(pModule != NULL) {
    // Import python function
    _pFunc = PyObject_GetAttrString(pModule, m_cfg.inputFuncName.c_str());
    Py_DECREF(pModule);
    if (_pFunc && PyCallable_Check(_pFunc)) {
      is_ready = true;
    }
  } else {
    Py_DECREF(pModule);
  }
  if (is_ready) {
    ACTS_INFO("ML-based track finding is ready.");
  } else {
    PyErr_Print();
    throw std::runtime_error("Failed to load track finding function");
  }
  return is_ready;
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
  // creating a thread state data structure,
  // storing thread state pointer
  // PyGILState_STATE gstate;
  // gstate = PyGILState_Ensure();

  // Read input data
  const auto& spacepoints =
    ctx.eventStore.get<SimSpacePointContainer>(m_cfg.inputSpacePoints);

  // Convert Input data to a list of size [num_measurements x measurement_features]
  size_t num_spacepoints = spacepoints.size();
  ACTS_INFO("Received " << num_spacepoints << " spacepoints");
  // input info are [idx, r, phi, z, 'cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
  size_t num_features = 13; // <TODO> move this to the configuration
  // std::vector<float> inputMatrix(num_spacepoints * num_features, 0.0);
  // // <TODO> Make configurable which sp features to use.
  // // vectorization is applied automatically?
  // // <ERROR> we use <r, phi, z> as inputs!
  // for(size_t idx=0; idx < num_spacepoints; ++idx){
  //   auto sp = spacepoints[idx];
  //   inputMatrix[num_features*idx] = idx;
  //   // cluster position
  //   inputMatrix[num_features*idx+1] = sp.x();
  //   inputMatrix[num_features*idx+2] = sp.y();
  //   inputMatrix[num_features*idx+3] = sp.z();
  //   // cluster shape uses default values for now
  // }

  std::vector<float> inputMatrix{
    21215, 0.0321, 0.8898, -0.0026, 2.0000, 0.3177, 1.6235, 1.1526, 0.0500, 0.1125, 0.3000, 0.3619, 2.8577,
    21254, 0.0333, 0.8894, -0.0028, 5.0000, 0.2932, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.6099,
    29500, 0.0716, 0.8774, -0.0079, 4.0000, 0.2621, 1.2490, 0.6435, 0.1500, 0.1125, 0.3000, 0.3294, -3.0288,
    29527, 0.0735, 0.8768, -0.0082, 4.0000, 0.3184, 1.2490, 0.6435, 0.1500, 0.1125, 0.3000, 0.3294, -2.8325,
    36548, 0.1164, 0.8631, -0.0142, 4.0000, 0.3258, 1.1633, 0.2742, 0.2000, 0.0563, 0.3000, 0.1554, -2.8365,
    42951, 0.1714, 0.8462, -0.0223, 5.0000, 0.3242, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.9372,
    42964., 0.1734, 0.8456, -0.0226, 5.0000, 0.3406, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.8566,
    75014., 0.2608, 0.8188, -0.0354, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.8893,
    81389., 0.3625, 0.7872, -0.0508, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.7098,
    81403., 0.3564, 0.7891, -0.0496, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.8220,
    81905., 0.3653, 0.7863, -0.0510, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.7098,
    81920., 0.3593, 0.7882, -0.0510, 4.0000, 4.0000, 0.3924, 1.3102, 0.3200, 1.2000, 0.5000, 1.4532, 2.9438,
    88050., 0.5018, 0.7431, -0.0736, 4.0000, 4.0000, 0.3924, 1.3102, 0.3200, 1.2000, 0.5000, 1.4532, 2.7151,
    88051., 0.4961, 0.7450, -0.0724, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.9011,
    94093., 0.6620, 0.6866, -0.0988, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.6499,
    94097., 0.6564, 0.6887, -0.0988, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.7115,
    110072., 0.8173, 0.6219, -0.1300, 7.0000, 7.0000, 0.0646, 1.4932, 0.8400, 10.8000, 0.7000, 2.9859, 2.6896,
    114958., 1.0236, 0.4845, -0.1898, 14.0000, 14.0000, 0.0640, 1.4165, 1.6800, 10.8000, 0.7000, 2.4809, 2.5348,
    114959., 1.0195, 0.4892, -0.1898, 15.0000, 15.0000, 0.0639, 1.4056, 1.8000, 10.8000, 0.7000, 2.4224, 2.6000
  };

  // Convert C++ vector to Python list
  // <TODO> Check memory management
  PyObject *pArgs, *pValue;
  pArgs = PyTuple_New(1);
  pValue = PyList_New(inputMatrix.size());
  vector_to_pylist(inputMatrix, pValue);
  PyTuple_SetItem(pArgs, 0, pValue);

  // call ML-based track finding
  pValue = PyObject_CallObject(_pFunc, pArgs);

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
          pSP = PyList_GetItem(pTrk, j); // cannot fail
          if(!PyLong_Check(pSP)) continue; // Skip non-integers
          auto idx_sp = PyLong_AsLong(pSP);
          if (idx_sp == -1 && PyErr_Occurred()){
            // Integer too big to fit in a C long, bail out
            continue;
          }
          protoTrack.push_back(spacepoints[idx_sp].measurementIndex());
        }
      }
      protoTracks.push_back(protoTrack);
    }
  } else {
    ACTS_WARNING("ML-based Track finding failed");
    protoTracks.push_back(ProtoTrack());
  }
  Py_DECREF(pArgs);
  Py_DECREF(pValue);

  // PyGILState_Release(gstate);

  // Py_Finalize(); # there are no finalize function... where should I put this?
  ACTS_INFO("Created " << protoTracks.size() << " proto tracks");
  ctx.eventStore.add(m_cfg.outputProtoTracks, std::move(protoTracks));

  return ActsExamples::ProcessCode::SUCCESS;
}


ActsExamples::ProcessCode ActsExamples::TrackFindingMLBasedAlgorithm::execute() const 
{

  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue; // returned python function values
  wchar_t *program = Py_DecodeLocale("/usr/bin/python", NULL);
  // wchar_t *program = Py_DecodeLocale("TEST", NULL);
  if (program == NULL) {
    fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    exit(1);
  }
  Py_SetProgramName(program);
  std::cout << "after setting program name" << std::endl;

  // Initialize Python session
  Py_Initialize();

  PySys_SetArgv(0, (wchar_t**)"dummy");
  printf("after setting argv\n");
  PyRun_SimpleString("import torch\n"
                     "print('torch version:',torch.__version__)\n");

  // Import Exatrkx Python module
  pName = PyUnicode_FromString("inference_fn");
  if (pName == NULL){
    PyErr_Print();
    throw std::runtime_error("String to unicode conversion failed");
  }
  pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if(pModule != NULL) {
    // Import python function
    pFunc = PyObject_GetAttrString(pModule, "gnn_track_finding");
    if (pFunc && PyCallable_Check(pFunc)) {
        std::cout << "function is callable" << std::endl;

        pArgs = PyTuple_New(1);
        std::vector<float> hids {
          21215, 0.0321, 0.8898, -0.0026, 2.0000, 0.3177, 1.6235, 1.1526, 0.0500, 0.1125, 0.3000, 0.3619, 2.8577,
          21254, 0.0333, 0.8894, -0.0028, 5.0000, 0.2932, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.6099,
          29500, 0.0716, 0.8774, -0.0079, 4.0000, 0.2621, 1.2490, 0.6435, 0.1500, 0.1125, 0.3000, 0.3294, -3.0288,
          29527, 0.0735, 0.8768, -0.0082, 4.0000, 0.3184, 1.2490, 0.6435, 0.1500, 0.1125, 0.3000, 0.3294, -2.8325,
          36548, 0.1164, 0.8631, -0.0142, 4.0000, 0.3258, 1.1633, 0.2742, 0.2000, 0.0563, 0.3000, 0.1554, -2.8365,
          42951, 0.1714, 0.8462, -0.0223, 5.0000, 0.3242, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.9372,
          42964., 0.1734, 0.8456, -0.0226, 5.0000, 0.3406, 1.0829, 0.5124, 0.2000, 0.1125, 0.3000, 0.3072, -2.8566,
          75014., 0.2608, 0.8188, -0.0354, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.8893,
          81389., 0.3625, 0.7872, -0.0508, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.7098,
          81403., 0.3564, 0.7891, -0.0496, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.8220,
          81905., 0.3653, 0.7863, -0.0510, 3.0000, 3.0000, 0.3980, 1.3734, 0.2400, 1.2000, 0.5000, 1.5145, 2.7098,
          81920., 0.3593, 0.7882, -0.0510, 4.0000, 4.0000, 0.3924, 1.3102, 0.3200, 1.2000, 0.5000, 1.4532, 2.9438,
          88050., 0.5018, 0.7431, -0.0736, 4.0000, 4.0000, 0.3924, 1.3102, 0.3200, 1.2000, 0.5000, 1.4532, 2.7151,
          88051., 0.4961, 0.7450, -0.0724, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.9011,
          94093., 0.6620, 0.6866, -0.0988, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.6499,
          94097., 0.6564, 0.6887, -0.0988, 5.0000, 5.0000, 0.3857, 1.2490, 0.4000, 1.2000, 0.5000, 1.3859, 2.7115,
          110072., 0.8173, 0.6219, -0.1300, 7.0000, 7.0000, 0.0646, 1.4932, 0.8400, 10.8000, 0.7000, 2.9859, 2.6896,
          114958., 1.0236, 0.4845, -0.1898, 14.0000, 14.0000, 0.0640, 1.4165, 1.6800, 10.8000, 0.7000, 2.4809, 2.5348,
          114959., 1.0195, 0.4892, -0.1898, 15.0000, 15.0000, 0.0639, 1.4056, 1.8000, 10.8000, 0.7000, 2.4224, 2.6000
        };
        pValue = PyList_New(hids.size());
        vector_to_pylist(hids, pValue);
        PyTuple_SetItem(pArgs, 0, pValue);
        pValue = PyObject_CallObject(pFunc, pArgs);

        if (pValue != NULL && PyList_Check(pValue)) {
          printf("Returned result is a list\n");
          Py_ssize_t num_trks = PyList_Size(pValue);
          printf("In total %ld tracks\n", num_trks);
          PyObject *pTrk, *pSP;
          for(Py_ssize_t i=0; i < num_trks; ++i){
            printf("track %ld:", i);
            pTrk = PyList_GetItem(pValue, i);
            if (pTrk && PyList_Check(pTrk)){
              Py_ssize_t num_sps = PyList_Size(pTrk);
              for(Py_ssize_t j=0; j < num_sps; ++j) {
                pSP = PyList_GetItem(pTrk, j);
                printf(" %ld", PyLong_AsLong(pSP));
              }
              printf(".\n");
            }
          }
          Py_DECREF(pValue);
        }
    }
  } else {
      PyErr_Print();
      throw std::runtime_error("Failed to load track finding function");
  }

  Py_Finalize();

  return ActsExamples::ProcessCode::SUCCESS;
}