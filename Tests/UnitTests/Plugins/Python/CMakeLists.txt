find_package (Python3 COMPONENTS Interpreter Development)
message( "Python3_FOUND = " ${Python3_FOUND})
message( "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message( "Python3_LIBRARIES = ${Python3_LIBRARIES}")



# set(unittest_extra_libraries ActsPythonPlugin)
add_executable(InvokePyFunc invoke_py_func.cpp)
target_include_directories(InvokePyFunc PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${Python3_INCLUDE_DIRS})
target_link_libraries(InvokePyFunc PUBLIC ${Python3_LIBRARIES})



#add_executable(PyInTbb py_in_tbb.cpp)