#pragma once
// Host-only parsing and span checks; no model, GPU, or allocation-size defaults.
#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kda_capture {
inline uint64_t positive(const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::invalid_argument("Expected a positive integer");
  const auto value = std::stoull(text);
  if (!value || value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    throw std::invalid_argument("Positive integer out of range");
  return value;
}
inline std::string required(const char* key) {
  const char* value = std::getenv(key);
  if (!value || !*value) throw std::invalid_argument(std::string("Missing ") + key);
  return value;
}
inline std::vector<std::string> split(const std::string& text, char separator) {
  std::vector<std::string> parts;
  size_t start = 0;
  do {
    const auto end = text.find(separator, start);
    parts.push_back(text.substr(start, end == std::string::npos ? end : end - start));
    if (end == std::string::npos) break;
    start = end + 1;
  } while (true);
  return parts;
}
inline std::vector<std::array<int64_t, 4>> shapes(const std::string& text) {
  std::vector<std::array<int64_t, 4>> result;
  for (const auto& row : split(text, ';')) {
    const auto cells = split(row, ',');
    if (cells.size() != 4) throw std::invalid_argument("Expected batch,M,K,N");
    std::array<int64_t, 4> shape;
    for (size_t i = 0; i != 4; ++i) shape[i] = positive(cells[i]);
    result.push_back(shape);
  }
  return result;
}
inline size_t span_bytes(const std::vector<int64_t>& shape,
                         const std::vector<int64_t>& stride, size_t limit) {
  if (shape.size() != stride.size() || shape.empty() || limit < sizeof(float))
    throw std::invalid_argument("Invalid tensor span contract");
  const size_t max_elements = limit / sizeof(float);
  size_t span = 1;
  for (size_t i = 0; i != shape.size(); ++i) {
    if (shape[i] <= 0 || stride[i] < 0) throw std::invalid_argument("Unsupported shape/stride");
    const uint64_t extent = shape[i] - 1;
    const uint64_t step = stride[i];
    if (step && extent > (max_elements - span) / step)
      throw std::length_error("Capture exceeds reviewed byte limit");
    span += extent * step;
  }
  return span * sizeof(float);
}
struct Config {
  std::vector<std::array<int64_t, 4>> allowed = shapes(required("KDA_CAPTURE_BMM_SHAPES"));
  size_t max_cases = positive(required("KDA_CAPTURE_MAX_CASES"));
  size_t max_bytes = positive(required("KDA_CAPTURE_MAX_BYTES"));
};
}
