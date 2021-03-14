find_package (Python3 COMPONENTS Interpreter Development)
message( "Python3_FOUND = " ${Python3_FOUND})
message( "Python3_INCLUDE_DIRS = ${Python3_INCLUDE_DIRS}")
message( "Python3_LIBRARIES = ${Python3_LIBRARIES}")

add_library(
  ActsExamplesTrackFindingMLBased SHARED
  src/TrackFindingMLBasedAlgorithm.cpp)

target_include_directories(
  ActsExamplesTrackFindingMLBased
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${Python3_INCLUDE_DIRS})

target_link_libraries(
  ActsExamplesTrackFindingMLBased
  PUBLIC
    ActsCore
    ActsExamplesFramework ${Python3_LIBRARIES})

install(
  TARGETS ActsExamplesTrackFindingMLBased
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})