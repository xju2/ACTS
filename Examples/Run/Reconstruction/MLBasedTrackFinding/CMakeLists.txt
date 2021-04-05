add_executable(
  ActsExampleTrackFindingMLBased
  TrackFindingMLBasedExample.cpp)
target_link_libraries(
  ActsExampleTrackFindingMLBased
  PRIVATE
    ActsExamplesFramework
    ActsExamplesCommonS
    ActsExamplesIoCsv
    ActsExamplesIoJson
    ActsExamplesIoRoot
    ActsExamplesDetectorGeneric
    ActsExamplesDigitization
    ActsExamplesMagneticField
    ActsExamplesTrackFinding # spacepoint maker should not live here!
    ActsExamplesTrackFindingMLBased
    ActsExamplesTrackFitting
  )


install(
  TARGETS
  ActsExampleTrackFindingMLBased
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


find_package (Python3 COMPONENTS Interpreter Development)

add_executable(
  ActsExampleTrackFindingMLBasedSimple
  TrackFindingMLBasedExampleSimple.cpp)

target_link_libraries(
  ActsExampleTrackFindingMLBasedSimple
  PUBLIC
  ${Python3_LIBRARIES}
  ROOT::Core ROOT::Hist
  PRIVATE Boost::filesystem dfelibs TBB::tbb
  ActsExamplesFramework
  ActsExamplesCommonS
  ActsExamplesIoCsv
  ActsExamplesIoJson
  ActsExamplesIoRoot
  ActsExamplesDetectorGeneric
  ActsExamplesDigitization
  ActsExamplesMagneticField
  ActsExamplesTrackFinding # spacepoint maker should not live here!
  ActsExamplesTrackFindingMLBased
  ActsExamplesTrackFitting
)
target_include_directories(
  ActsExampleTrackFindingMLBasedSimple
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> ${Python3_INCLUDE_DIRS}
)
target_compile_features(
  ActsExampleTrackFindingMLBasedSimple
  PUBLIC cxx_std_17)


install(
  TARGETS
  ActsExampleTrackFindingMLBasedSimple
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})