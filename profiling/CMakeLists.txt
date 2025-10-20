cmake_minimum_required(VERSION 3.14)  # Required for FetchContent

# Set CUDA compiler path before project declaration
set(CMAKE_CUDA_COMPILER /opt/nvidia/hpc_sdk/Linux_x86_64/25.9/compilers/bin/nvcc)

# Initialize project with just CXX language first
project(parrot_profiling LANGUAGES CXX)

# Include FetchContent module for downloading dependencies
include(FetchContent)

# Fetch the official parrot library from NVlabs/parrot
FetchContent_Declare(
  parrot
  GIT_REPOSITORY https://github.com/NVlabs/parrot.git
  GIT_TAG main  # You can specify a specific tag/commit if needed
  GIT_SHALLOW TRUE
)

# Make parrot available
FetchContent_MakeAvailable(parrot)

# Configure ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
    # Set ccache as the launcher for both C++ and CUDA
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
    message(STATUS "ccache not found - compile performance may be reduced")
endif()

# CUDA Architecture Configuration
# Option 1: User can specify via -DCUDA_ARCH=75,89
# Option 2: Auto-detect from current GPU
# Option 3: Build for common architectures
set(CUDA_ARCH "AUTO" CACHE STRING "CUDA architectures to build for (e.g., '75', '75,89', 'AUTO', 'ALL')")

# Now enable CUDA language
enable_language(CUDA)

# Function to detect GPU architecture using nvidia-smi
function(detect_gpu_architecture output_var)
    # Try using nvidia-smi first (faster and more reliable)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE nvidia_smi_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(nvidia_smi_output)
            # Take the first GPU's compute capability
            string(REGEX MATCH "^[0-9]+\\.[0-9]+" first_gpu_cap "${nvidia_smi_output}")
            if(first_gpu_cap)
                string(REPLACE "." "" arch_output "${first_gpu_cap}")
                set(${output_var} "${arch_output}" PARENT_SCOPE)
                message(STATUS "Detected GPU architecture via nvidia-smi: ${arch_output}")
                return()
            endif()
        endif()
    endif()
    
    # Fallback: try using deviceQuery if available
    find_program(DEVICE_QUERY deviceQuery PATHS
        /usr/local/cuda/extras/demo_suite
        /opt/cuda/extras/demo_suite
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../extras/demo_suite
    )
    
    if(DEVICE_QUERY)
        execute_process(
            COMMAND ${DEVICE_QUERY}
            OUTPUT_VARIABLE device_query_output
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(device_query_output)
            string(REGEX MATCH "CUDA Capability Major/Minor version number:[^0-9]*([0-9]+)\\.([0-9]+)" 
                   match_result "${device_query_output}")
            if(CMAKE_MATCH_1 AND CMAKE_MATCH_2)
                set(arch_output "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
                set(${output_var} "${arch_output}" PARENT_SCOPE)
                message(STATUS "Detected GPU architecture via deviceQuery: ${arch_output}")
                return()
            endif()
        endif()
    endif()
    
    # If all detection methods fail
    message(WARNING "Failed to detect GPU architecture - no nvidia-smi or deviceQuery found")
    set(${output_var} "" PARENT_SCOPE)
endfunction()

# Set CUDA architectures based on user choice
if(CUDA_ARCH STREQUAL "AUTO")
    detect_gpu_architecture(detected_arch)
    if(detected_arch)
        set(CMAKE_CUDA_ARCHITECTURES ${detected_arch})
        message(STATUS "Using auto-detected CUDA architecture: ${detected_arch}")
    else()
        # Fallback to common architectures if detection fails
        set(CMAKE_CUDA_ARCHITECTURES "75;89")
        message(STATUS "Auto-detection failed. Building for architectures: 75 (Turing/RTX), 89 (Ampere)")
    endif()
elseif(CUDA_ARCH STREQUAL "ALL")
    # Build for common modern architectures
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    message(STATUS "Building for all common CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    # Use user-specified architectures
    string(REPLACE "," ";" CUDA_ARCH_LIST ${CUDA_ARCH})
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
    message(STATUS "Using user-specified CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler timing flags for performance analysis
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftime-report")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --time")

# CUDA settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Add parrot include directory from FetchContent
FetchContent_GetProperties(parrot)
if(parrot_POPULATED)
    include_directories(${parrot_SOURCE_DIR})
    message(STATUS "Using parrot library from: ${parrot_SOURCE_DIR}")
    
    # Get CCCL from parrot's build and prioritize it over system headers
    # This ensures we use Thrust 3.2.0+ instead of system Thrust 3.0.1
    FetchContent_GetProperties(cccl)
    if(cccl_POPULATED)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR})
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/cub)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/thrust)
        include_directories(BEFORE SYSTEM ${cccl_SOURCE_DIR}/libcudacxx/include)
        
        # Add compiler flags to prioritize CCCL headers
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/cub")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/thrust")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -isystem ${cccl_SOURCE_DIR}/libcudacxx/include")
        
        message(STATUS "Using CCCL (Thrust 3.2.0+) from: ${cccl_SOURCE_DIR}")
    endif()
endif()

# Function to create CUDA executable with proper settings and output directory
function(add_cuda_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    set_target_properties(${target_name} PROPERTIES 
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    
    # Set output directory to the same directory as the source file
    get_filename_component(source_dir ${source_file} DIRECTORY)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${source_dir}")
endfunction()

# Add profiling executables

# AresDB Expand examples
add_cuda_executable(expand_thrust_cu aresdb_expand/expand.cu)
add_cuda_executable(expand_parrot_cu aresdb_expand/expand_parrot.cu)

# Outer Sum examples
add_cuda_executable(outer_sum_basic_cu outer_sum/outer_sum.cu)
add_cuda_executable(outer_sum_thrust_cu outer_sum/outer_sum_thrust.cu)

# Paddle Mode examples  
add_cuda_executable(paddle_mode_basic_cu paddle_mode/paddle_paddle_mode.cu)
add_cuda_executable(parrot_mode_cu paddle_mode/parrot_mode.cu)

# SF2 example
add_cuda_executable(sf2_parrot_cu sf2/sf2_parrot_cpp_cu.cu)

# Print summary of targets
message(STATUS "Configured profiling targets:")
message(STATUS "  - expand_thrust_cu: AresDB expand with Thrust implementation")
message(STATUS "  - expand_parrot_cu: AresDB expand with Parrot implementation")
message(STATUS "  - outer_sum_basic_cu: Basic outer sum implementation")
message(STATUS "  - outer_sum_thrust_cu: Thrust-based outer sum implementation")
message(STATUS "  - paddle_mode_basic_cu: PaddlePaddle mode computation")
message(STATUS "  - parrot_mode_cu: Parrot-based mode computation")
message(STATUS "  - sf2_parrot_cu: Sushi For Two problem using parrot")

# Use all cores for fast building (per BUILDING.md instruction)
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${N})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${N})
    message(STATUS "Building with ${N} cores for maximum speed")
endif()
