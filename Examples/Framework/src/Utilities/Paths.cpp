// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "ActsExamples/Utilities/Paths.hpp"

#include "Acts/Utilities/Logger.hpp"

#include <charconv>
#include <cstdio>
#include <regex>
#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>

std::string ActsExamples::ensureWritableDirectory(const std::string& dir) {
  using boost::filesystem::current_path;
  using boost::filesystem::path;

  auto dir_path = dir.empty() ? current_path() : path(dir);
  if (exists(dir_path) and not is_directory(dir_path)) {
    throw std::runtime_error("'" + dir +
                             "' already exists but is not a directory");
  }
  create_directories(dir_path);
  return canonical(dir_path).native();
}

std::string ActsExamples::joinPaths(const std::string& dir,
                                    const std::string& name) {
  if (dir.empty()) {
    return name;
  } else {
    return dir + '/' + name;
  }
}

std::string ActsExamples::perEventFilepath(const std::string& dir,
                                           const std::string& name,
                                           size_t event) {
  char prefix[64];

  snprintf(prefix, sizeof(prefix), "event%09zu-", event);

  if (dir.empty()) {
    return prefix + name;
  } else {
    return dir + '/' + prefix + name;
  }
}

std::pair<size_t, size_t> ActsExamples::determineEventFilesRange(
    const std::string& dir, const std::string& name) {
  using boost::filesystem::current_path;
  using boost::filesystem::directory_iterator;
  using boost::filesystem::path;

  ACTS_LOCAL_LOGGER(
      Acts::getDefaultLogger("EventFilesRange", Acts::Logging::VERBOSE));

  // ensure directory path is valid
  auto dir_path = dir.empty() ? current_path() : path(dir);
  if (not exists(dir_path)) {
    throw std::runtime_error("'" + dir_path.native() + "' does not exists");
  }
  if (not is_directory(dir_path)) {
    throw std::runtime_error("'" + dir_path.native() + "' is not a directory");
  }

  // invalid default range that allows simple restriction later on
  size_t eventMin = SIZE_MAX;
  size_t eventMax = 0;

  // filter matching event files from the directory listing
  std::string filename;
  std::regex re("^event([0-9]+)-" + name + "$");
  // std::cmatch match;

  for (const auto& f : directory_iterator(dir_path)) {
    if ((not exists(f.status())) or (not is_regular_file(f.status()))) {
      continue;
    }
    // keep a copy so the match can refer to the underlying const char*
    filename = f.path().filename().native();
    if (filename.find(name) != std::string::npos) {
      std::string::size_type start_pos = 0;
      std::string::size_type end_pos = 0;
      for(std::string::size_type idx=5; idx < filename.length(); idx++){
        if(filename[idx] != '0' && start_pos < 1) start_pos = idx;
        if(filename[idx] == '-' && end_pos < 1) {
          end_pos = idx; break;
        }
      }
      size_t event = 0;

      int length = end_pos-start_pos;
      if (length == 0) {
        length = 1;
        start_pos --;
      }
      event = std::stol(filename.substr(start_pos, length));

      // enlarge available event range
      eventMin = std::min(eventMin, event);
      eventMax = std::max(eventMax, event);
    }
  }
  ACTS_VERBOSE("Detected event range [" << eventMin << "," << eventMax << "]");

  // should only occur if no files matched and the initial values persisted.
  if (eventMax < eventMin) {
    return std::make_pair(0u, 0u);
  }
  return std::make_pair(eventMin, eventMax + 1);
}
