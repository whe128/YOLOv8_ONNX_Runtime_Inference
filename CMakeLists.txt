

cmake_minimum_required(VERSION 3.18)

project(YOLOv8Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory")
# CUDA DLL
set(CUDA_DLL_ROOT "" CACHE PATH "CUDA DLL directory")
# cuDNN DLL
set(CUDNN_DLL_ROOT "" CACHE PATH "cuDNN DLL directory")

find_path(ONNXRUNTIME_INCLUDE_DIR
    onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED
)


find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED
)

add_executable(yolov8
    src/inference.cpp
)

add_executable(yolov8_stream
    src/inference_onnx_stream.cpp
)

target_include_directories(yolov8 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_include_directories(yolov8_stream PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(yolov8 PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

target_link_libraries(yolov8_stream PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

option(ENABLE_CUDA "Use the ONNX Runtime CUDA execution provider" OFF)
if(ENABLE_CUDA)
    target_compile_definitions(yolov8_stream PRIVATE USE_CUDA)
endif()

# Copy ONNX Runtime DLL to executable directory on Windows
if(WIN32)
    # ONNX Runtime
    add_custom_command(TARGET yolov8 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"

        "$<TARGET_FILE_DIR:yolov8>"
    )

    # CUDA Runtime DLLs
    file(GLOB CUDA_DLLS "${CUDA_DLL_ROOT}/*.dll")
    foreach(DLL ${CUDA_DLLS})
        add_custom_command(TARGET yolov8 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}"
            "$<TARGET_FILE_DIR:yolov8>"
        )
    endforeach()

    # cuDNN DLLs
    file(GLOB CUDNN_DLLS "${CUDNN_DLL_ROOT}/*.dll")
    foreach(DLL ${CUDNN_DLLS})
        add_custom_command(TARGET yolov8 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}"
            "$<TARGET_FILE_DIR:yolov8>"
        )
    endforeach()

    # OpenCV
    add_custom_command(TARGET yolov8 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:opencv_world>"
        "$<TARGET_FILE_DIR:yolov8>"
    )

    get_filename_component(OPENCV_BIN_DIR
        "${OpenCV_DIR}/../bin"
        ABSOLUTE
    )
    file(GLOB OPENCV_VIDEOIO_PLUGINS
        "${OPENCV_BIN_DIR}/opencv_videoio_*.dll"
    )
    foreach(PLUGIN ${OPENCV_VIDEOIO_PLUGINS})
        add_custom_command(TARGET yolov8 POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PLUGIN}"
            "$<TARGET_FILE_DIR:yolov8>"
        )
    endforeach()

    add_custom_command(TARGET yolov8_stream POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:yolov8_stream>"
    )

    add_custom_command(TARGET yolov8_stream POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:opencv_world>"
        "$<TARGET_FILE_DIR:yolov8_stream>"
    )

    foreach(PLUGIN ${OPENCV_VIDEOIO_PLUGINS})
        add_custom_command(TARGET yolov8_stream POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PLUGIN}"
            "$<TARGET_FILE_DIR:yolov8_stream>"
        )
    endforeach()

endif()

# cmake -S . -B build -DONNXRUNTIME_ROOT="C:/environment/onnxruntime-win-x64-gpu_cuda13-1.31.1"  -DOpenCV_DIR="C:/environment/opencv/build/x64/vc16/lib" -DCUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"

# cmake --build build --config Release -j 4
# cmake -S . -B build -DONNXRUNTIME_ROOT="C:/environment/onnxruntime-win-x64-gpu_cuda13-1.31.1" -DCUDA_DLL_ROOT="C:/environment/Python313/Lib/site-packages/nvidia/cu13/bin/x86_64" -DCUDNN_DLL_ROOT="C:/environment/Python313/Lib/site-packages/nvidia/cudnn/bin" -DOpenCV_DIR="C:/environment/opencv/build/x64/vc16/lib"


# cmake -S . -B build  -DOpenCV_DIR="C:/environment/opencv/build/x64/vc16/lib"  -DONNXRUNTIME_ROOT="C:/environment/onnxruntime-win-x64-gpu_cuda13-1.31.1" -DENABLE_CUDA=ON

# cmake --build build --config Release -j 4
