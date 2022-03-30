add_library(
  ActsPluginExaTrkX SHARED
  src/ExaTrkXTrackFinding.cpp)

# set_target_properties(ActsPluginExaTrkX PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# set(CUGRAPH_CUDA_FLAGS "")
# list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
# list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
# list(APPEND CUGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)
# if(CMAKE_BUILD_TYPE MATCHES Debug)
#     message(STATUS "Building with debugging flags")
#     list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
# endif(CMAKE_BUILD_TYPE MATCHES Debug)

# set(CUGRAPH_CXX_FLAGS "")
# if(CMAKE_COMPILER_IS_GNUCXX)
#     list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
# endif(CMAKE_COMPILER_IS_GNUCXX)

# target_compile_options(ActsPluginExaTrkX
#     PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
#             "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>")

target_include_directories(
  ActsPluginExaTrkX
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_include_directories(
  ActsPluginExaTrkX
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(
  ActsPluginExaTrkX
  PRIVATE
    frnn
    cugraph
)

target_link_libraries(
  ActsPluginExaTrkX
  PUBLIC
    ${TORCH_LIBRARIES}
)

install(
  TARGETS ActsPluginExaTrkX
  EXPORT ActsPluginExaTrkXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(
    DIRECTORY include/Acts
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
