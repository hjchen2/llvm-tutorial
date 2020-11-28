cmake_minimum_required(VERSION 3.1)

project(toy)

set(LLVM_ROOT /home/chenhoujiang/nmt/llvm-project)
set(LLVM_INCLUDE_DIR
    ${LLVM_ROOT}/llvm/include
    ${LLVM_ROOT}/build/include)
set(LLVM_LIBRARIES
    ${LLVM_ROOT}/build/lib/libLLVMSupport.a
    ${LLVM_ROOT}/build/lib/libLLVMOption.a
)

set(CMAKE_CXX_FLAGS "-std=c++14")

file(GLOB TOY_SRCS *.cc)
add_executable(toy ${TOY_SRCS})

include_directories(${LLVM_INCLUDE_DIR})
target_link_libraries(toy PUBLIC ${LLVM_LIBRARIES} -ltinfo -pthread)
