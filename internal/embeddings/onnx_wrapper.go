package embeddings

/*
#cgo CFLAGS: -I${SRCDIR} -I${SRCDIR}/../../onnxruntime-osx-arm64-1.14.0/include
#cgo LDFLAGS: -L${SRCDIR}/../../onnxruntime-osx-arm64-1.14.0/lib -lonnxruntime -Wl,-rpath,${SRCDIR}/../../onnxruntime-osx-arm64-1.14.0/lib

// Ensure our version is used by undefining any existing version first
#ifdef ORT_API_VERSION
#undef ORT_API_VERSION
#endif

// Include our API version header first to set the version
#include "onnx_api_version.h"

// Then include the ONNX runtime headers
#include "onnxruntime_c_api.h"

// Verify the version is set correctly
#if ORT_API_VERSION != 14
#error "ORT_API_VERSION must be 14"
#endif
*/
import "C"
