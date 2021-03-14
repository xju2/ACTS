#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Python.h"

void vector_to_pylist(std::vector<int>& hids, PyObject* py_hids) {
  // py_hids = (PyListObject*) PyList_New(0);
  PyObject* pHid;
  for(int i=0; i < (int) hids.size(); ++i){
    std::cout << "hit id: " << hids.at(i) << std::endl;
    pHid = PyLong_FromLong(hids.at(i));
    if (!pHid) {
      fprintf(stderr, "Cannot convert argument\n");
      break;
    }
    PyList_SetItem(py_hids, i, pHid);
  }
}

int main(int argc, char** argv){
  PyObject *pName, *pModule, *pFunc;
  PyObject *pArgs, *pValue; // returned python function values

  // Initialize Python session
  Py_Initialize();

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
        std::vector<int> hids {1, 2, 3, 4, 5};
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

