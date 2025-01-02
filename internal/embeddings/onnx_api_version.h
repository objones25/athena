#ifndef ONNX_API_VERSION_H
#define ONNX_API_VERSION_H

// Undefine any existing version
#ifdef ORT_API_VERSION
#undef ORT_API_VERSION
#endif

// Override the API version to match what's supported by ONNX Runtime 1.14.0
#define ORT_API_VERSION 14

#endif // ONNX_API_VERSION_H 