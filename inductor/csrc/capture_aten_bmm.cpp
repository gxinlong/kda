// Capture explicitly reviewed FP32 BMM boundaries through the AOTI C ABI.
// Diagnostic worker only: synchronization and host copies invalidate timing.
#include <cuda.h>
#include "capture_bmm_config.h"
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

struct AtenTensorOpaque;
using Handle = AtenTensorOpaque*;
using Error = int32_t;

namespace {
std::mutex mutex;
size_t count = 0;

void* torch_symbol(const char* name) {
  if (void* found = dlsym(RTLD_NEXT, name)) return found;
  for (const char* library : {"libtorch_cuda.so", "libtorch_cpu.so"}) {
    void* handle = dlopen(library, RTLD_NOW | RTLD_NOLOAD);
    if (handle) if (void* found = dlsym(handle, name)) return found;
  }
  std::fprintf(stderr, "KDA BMM capture: unresolved %s: %s\n", name, dlerror());
  std::abort();
}

void* cuda_symbol(const char* name) {
  static void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  void* found = driver ? dlsym(driver, name) : nullptr;
  if (!found) std::abort();
  return found;
}

void check(int result) {
  if (result) {
    std::fprintf(stderr, "KDA BMM capture: API error %d\n", result);
    std::abort();
  }
}

struct Tensor {
  std::vector<int64_t> shape, stride;
  int32_t dtype = -1;
  CUdeviceptr pointer = 0, allocation_base = 0;
  size_t bytes = 0, allocation_bytes = 0;
};

Tensor inspect(Handle tensor) {
  using Dim = Error(*)(Handle, int64_t*);
  using Array = Error(*)(Handle, int64_t**);
  using Dtype = Error(*)(Handle, int32_t*);
  using Pointer = Error(*)(Handle, void**);
  Tensor out;int64_t dim=0;int64_t* shape=nullptr;int64_t* stride=nullptr;void* pointer=nullptr;
  check(reinterpret_cast<Dim>(torch_symbol("aoti_torch_get_dim"))(tensor,&dim));
  check(reinterpret_cast<Array>(torch_symbol("aoti_torch_get_sizes"))(tensor,&shape));
  check(reinterpret_cast<Array>(torch_symbol("aoti_torch_get_strides"))(tensor,&stride));
  check(reinterpret_cast<Dtype>(torch_symbol("aoti_torch_get_dtype"))(tensor,&out.dtype));
  check(reinterpret_cast<Pointer>(torch_symbol("aoti_torch_get_data_ptr"))(tensor,&pointer));
  out.shape.assign(shape,shape+dim);out.stride.assign(stride,stride+dim);
  out.pointer=reinterpret_cast<uintptr_t>(pointer);return out;
}

void record(std::ostream& out, const Tensor& tensor) {
  out << "{\"shape\":[";
  for (size_t i=0;i<tensor.shape.size();++i) { if(i)out<<',';out<<tensor.shape[i]; }
  out << "],\"stride\":[";
  for (size_t i=0;i<tensor.stride.size();++i) { if(i)out<<',';out<<tensor.stride[i]; }
  out << "],\"dtype\":\"torch.float32\",\"dtype_code\":" << tensor.dtype
      << ",\"bytes\":" << tensor.bytes << ",\"pointer\":" << tensor.pointer
      << ",\"allocation_base\":" << tensor.allocation_base
      << ",\"allocation_bytes\":" << tensor.allocation_bytes << '}';
}

void prepare(Tensor& tensor, size_t limit) {
  tensor.bytes=kda_capture::span_bytes(tensor.shape,tensor.stride,limit);
  using Range=CUresult(CUDAAPI*)(CUdeviceptr*,size_t*,CUdeviceptr);
  check(reinterpret_cast<Range>(cuda_symbol("cuMemGetAddressRange_v2"))(&tensor.allocation_base,&tensor.allocation_bytes,tensor.pointer));
  if(tensor.pointer<tensor.allocation_base || tensor.pointer-tensor.allocation_base>tensor.allocation_bytes ||
     tensor.bytes>tensor.allocation_bytes-(tensor.pointer-tensor.allocation_base))std::abort();
}

void copy(const std::filesystem::path& path,const Tensor& tensor) {
  using Copy=CUresult(CUDAAPI*)(void*,CUdeviceptr,size_t);
  std::vector<char> data(tensor.bytes);
  check(reinterpret_cast<Copy>(cuda_symbol("cuMemcpyDtoH_v2"))(data.data(),tensor.pointer,data.size()));
  std::ofstream file(path,std::ios::binary|std::ios::trunc);
  file.write(data.data(),data.size());if(!file)std::abort();
}
}

extern "C" Error aoti_torch_cuda_bmm_out(Handle output,Handle left,Handle right) {
  using Bmm=Error(*)(Handle,Handle,Handle);
  static auto real=reinterpret_cast<Bmm>(torch_symbol("aoti_torch_cuda_bmm_out"));
  std::unique_lock<std::mutex> guard(mutex);
  static const kda_capture::Config config;
  if(count>=config.max_cases){guard.unlock();return real(output,left,right);}
  Tensor a=inspect(left),b=inspect(right),c=inspect(output);
  using Dtype=int32_t(*)();
  const auto fp32=reinterpret_cast<Dtype>(torch_symbol("aoti_torch_dtype_float32"))();
  bool match=false;
  if(a.shape.size()==3 && b.shape.size()==3 && a.dtype==fp32 && b.dtype==fp32 && c.dtype==fp32) {
    for(const auto& shape : config.allowed) {
      if(a.shape==std::vector<int64_t>({shape[0],shape[1],shape[2]}) &&
         b.shape==std::vector<int64_t>({shape[0],shape[2],shape[3]})) { match=true;break; }
    }
  }
  if(!match){guard.unlock();return real(output,left,right);}
  if(c.shape!=std::vector<int64_t>({a.shape[0],a.shape[1],b.shape[2]}))std::abort();
  const char* directory=std::getenv("KDA_CAPTURE_DIR");if(!directory)std::abort();
  std::filesystem::path root(directory),folder=root/("case_"+std::to_string(count));
  if(std::filesystem::exists(folder))std::abort();
  std::filesystem::create_directories(folder);
  using Stream=Error(*)(int32_t,void**);void* stream=nullptr;
  check(reinterpret_cast<Stream>(torch_symbol("aoti_torch_get_current_cuda_stream"))(0,&stream));
  using Sync=CUresult(CUDAAPI*)(CUstream);
  auto sync=reinterpret_cast<Sync>(cuda_symbol("cuStreamSynchronize"));
  check(sync(static_cast<CUstream>(stream)));
  prepare(a,config.max_bytes);prepare(b,config.max_bytes);prepare(c,config.max_bytes);copy(folder/"a.bin",a);copy(folder/"b.bin",b);
  Error result=real(output,left,right);check(result);check(sync(static_cast<CUstream>(stream)));
  copy(folder/"out.bin",c);
  std::ofstream metadata(folder/"metadata.json");metadata<<"{\"case_id\":"<<count<<",\"a\":";record(metadata,a);
  metadata<<",\"b\":";record(metadata,b);metadata<<",\"out\":";record(metadata,c);metadata<<"}\n";
  if(!metadata)std::abort();
  ++count;
  std::ofstream summary(root/"capture.json");summary<<"{\"kind\":\"aten_bmm\",\"cases\":[";
  for(size_t i=0;i<count;++i){if(i)summary<<',';summary<<"\"case_"<<i<<"\"";}summary<<"]}\n";
  if(!summary)std::abort();
  std::fprintf(stderr,"KDA captured BMM case %zu: B=%ld M=%ld N=%ld K=%ld\n",count-1,a.shape[0],a.shape[1],b.shape[2],a.shape[2]);
  return result;
}
