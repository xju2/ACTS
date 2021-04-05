#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Python.h"

void vector_to_pylist(std::vector<float>& hids, PyObject* py_hids) {
  // py_hids = (PyListObject*) PyList_New(0);
  PyObject* pHid;
  for(int i=0; i < (int) hids.size(); ++i){
    // std::cout << "hit id: " << hids.at(i) << std::endl;
    pHid = PyFloat_FromDouble(hids.at(i));
    if (!pHid) {
      fprintf(stderr, "Cannot convert argument\n");
      break;
    }
    PyList_SetItem(py_hids, i, pHid);
  }
}

int main(int argc, char *argv[])
{
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue; // returned python function values
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  // wchar_t *program = Py_DecodeLocale("TEST", NULL);
  if (program == NULL) {
    fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
    exit(1);
  }
  Py_SetProgramName(program);
  std::cout << "after setting program name" << std::endl;

  // Initialize Python session
  Py_Initialize();

  printf("%d argc, %s\n", argc, argv[0]);
  PySys_SetArgv(0, (wchar_t**)argv);
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
  return 0;
}

