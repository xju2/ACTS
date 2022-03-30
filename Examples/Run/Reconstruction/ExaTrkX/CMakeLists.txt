add_executable(
  ActsExampleExaTrkXGeneric
  TrackFindingExaTrkXGeneric.cpp)

target_link_libraries(
  ActsExampleExaTrkXGeneric
  PRIVATE
    ActsExamplesCommon
    ActsExamplesFramework
    ActsExamplesDigitization
    ActsExamplesDetectorGeneric
    ActsExamplesIoPerformance
    ActsExamplesTrackFindingExaTrkX
    # ActsPluginExaTrkX
    Boost::filesystem
    Boost::program_options
)

install(
  TARGETS
  ActsExampleExaTrkXGeneric
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
