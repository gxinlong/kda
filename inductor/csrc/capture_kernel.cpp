// Capture one named AOT driver launch. Use only in a separate validation worker.
#include <cuda.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
std::mutex names_mutex;
std::unordered_map<CUfunction, std::string> names;
std::atomic<bool> captured{false};

void* symbol(const char* name) {
  static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  void* result = driver ? dlsym(driver, name) : nullptr;
  if (!result) {
    std::fprintf(stderr, "KDA capture: cannot resolve %s\n", name);
    std::abort();
  }
  return result;
}

void check(CUresult result) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "KDA capture: CUDA error %d\n", static_cast<int>(result));
    std::abort();
  }
}

struct Argument {
  char role;
  size_t bytes;
  CUdeviceptr pointer = 0;
  CUdeviceptr base = 0;
  size_t allocation_bytes = 0;
};

void save(const std::filesystem::path& path, const void* data, size_t bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(static_cast<const char*>(data), bytes);
  if (!stream) {
    std::fprintf(stderr, "KDA capture: failed to write %s\n", path.c_str());
    std::abort();
  }
}
} // namespace

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
  using Function = CUresult(CUDAAPI*)(CUfunction*, CUmodule, const char*);
  static auto real = reinterpret_cast<Function>(symbol("cuModuleGetFunction"));
  CUresult result = real(function, module, name);
  if (result == CUDA_SUCCESS) {
    std::lock_guard<std::mutex> lock(names_mutex);
    names[*function] = name;
  }
  return result;
}

extern "C" CUresult CUDAAPI cuLaunchKernel(
    CUfunction function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream stream,
    void** parameters, void** extra) {
  using Launch = CUresult(CUDAAPI*)(CUfunction, unsigned, unsigned, unsigned,
      unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
  static auto real = reinterpret_cast<Launch>(symbol("cuLaunchKernel"));
  const char* target = std::getenv("KDA_CAPTURE_KERNEL");
  bool match = false;
  if (target && !captured.load()) {
    std::lock_guard<std::mutex> lock(names_mutex);
    auto found = names.find(function);
    match = found != names.end() && found->second == target;
  }
  if (!match || captured.exchange(true))
    return real(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);

  const char* directory = std::getenv("KDA_CAPTURE_DIR");
  const char* specification = std::getenv("KDA_CAPTURE_ARGS");
  if (!directory || !specification || !parameters || extra) {
    std::fprintf(stderr, "KDA capture: missing directory/ABI or unsupported extra launch arguments\n");
    std::abort();
  }
  using Sync = CUresult(CUDAAPI*)(CUstream);
  using Copy = CUresult(CUDAAPI*)(void*, CUdeviceptr, size_t);
  using Range = CUresult(CUDAAPI*)(CUdeviceptr*, size_t*, CUdeviceptr);
  auto sync = reinterpret_cast<Sync>(symbol("cuStreamSynchronize"));
  auto copy = reinterpret_cast<Copy>(symbol("cuMemcpyDtoH_v2"));
  auto range = reinterpret_cast<Range>(symbol("cuMemGetAddressRange_v2"));
  const std::filesystem::path root(directory);
  std::filesystem::create_directories(root);
  if (std::filesystem::exists(root / "capture.json")) {
    std::fprintf(stderr, "KDA capture: output already exists\n");
    std::abort();
  }
  check(sync(stream));
  std::vector<Argument> arguments;
  std::istringstream specs(specification);
  std::string token;
  while (std::getline(specs, token, ',')) {
    if (token.size() < 3 || token[1] != ':' || std::string("ios").find(token[0]) == std::string::npos)
      std::abort();
    Argument argument{token[0], std::stoull(token.substr(2))};
    const size_t index = arguments.size();
    if (argument.role == 's') {
      if (argument.bytes > 8) std::abort();
      save(root / ("arg" + std::to_string(index) + ".bin"), parameters[index], argument.bytes);
    } else {
      argument.pointer = *static_cast<CUdeviceptr*>(parameters[index]);
      check(range(&argument.base, &argument.allocation_bytes, argument.pointer));
      if (argument.pointer < argument.base || argument.pointer - argument.base > argument.allocation_bytes ||
          argument.bytes > argument.allocation_bytes - (argument.pointer - argument.base)) std::abort();
      if (argument.role == 'i') {
        std::vector<char> data(argument.bytes);
        check(copy(data.data(), argument.pointer, argument.bytes));
        save(root / ("arg" + std::to_string(index) + ".bin"), data.data(), data.size());
      }
    }
    arguments.push_back(argument);
  }
  CUresult result = real(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
  check(result);
  check(sync(stream));
  for (size_t index = 0; index < arguments.size(); ++index) {
    const auto& argument = arguments[index];
    if (argument.role == 'o') {
      std::vector<char> data(argument.bytes);
      check(copy(data.data(), argument.pointer, argument.bytes));
      save(root / ("arg" + std::to_string(index) + ".bin"), data.data(), data.size());
    }
  }
  std::ofstream metadata(root / "capture.json");
  metadata << "{\"kernel\":\"" << target << "\",\"grid\":[" << gx << ',' << gy << ',' << gz
           << "],\"block\":[" << bx << ',' << by << ',' << bz << "],\"shared_bytes\":" << shared
           << ",\"arguments\":[";
  for (size_t index = 0; index < arguments.size(); ++index) {
    const auto& argument = arguments[index];
    if (index) metadata << ',';
    metadata << "{\"role\":\"" << argument.role << "\",\"bytes\":" << argument.bytes
             << ",\"pointer\":" << argument.pointer << ",\"allocation_base\":" << argument.base
             << ",\"allocation_bytes\":" << argument.allocation_bytes << '}';
  }
  metadata << "]}\n";
  if (!metadata) std::abort();
  return result;
}
