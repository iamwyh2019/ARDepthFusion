import Accelerate
import CoreML
import CoreImage
import CoreVideo
import Vision

struct DepthMapData: Sendable {
    let values: [Float]
    let width: Int
    let height: Int
}

final class DepthEstimator: @unchecked Sendable {
    nonisolated static let shared = DepthEstimator()

    nonisolated let model: VNCoreMLModel
    private let lock = NSLock()
    nonisolated(unsafe) private var pendingContinuation: CheckedContinuation<DepthMapData?, Never>?

    nonisolated static func preload() async {
        await Task.detached(priority: .userInitiated) {
            _ = shared
        }.value
    }

    private init?() {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        guard let coreMLModel = try? DepthAnythingV2SmallF16(configuration: config).model,
              let vnModel = try? VNCoreMLModel(for: coreMLModel) else {
            print("Failed to load Depth Anything V2 model")
            return nil
        }
        self.model = vnModel
        print("Depth Anything V2 model loaded")
    }

    nonisolated func estimateDepth(cgImage: CGImage) async -> DepthMapData? {
        let start = CFAbsoluteTimeGetCurrent()

        let result: DepthMapData? = await withCheckedContinuation { continuation in
            lock.lock()
            pendingContinuation = continuation
            lock.unlock()

            let request = VNCoreMLRequest(model: model) { [self] request, error in
                let depthData = self.processResult(request: request, error: error)
                lock.lock()
                let cont = pendingContinuation
                pendingContinuation = nil
                lock.unlock()
                cont?.resume(returning: depthData)
            }
            request.imageCropAndScaleOption = .scaleFill

            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            do {
                try handler.perform([request])
            } catch {
                print("Depth estimation failed: \(error.localizedDescription)")
                lock.lock()
                let cont = pendingContinuation
                pendingContinuation = nil
                lock.unlock()
                cont?.resume(returning: nil)
            }
        }

        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        if result != nil {
            print(String(format: "Depth Anything inference: %.0fms", elapsed))
        }
        return result
    }

    private nonisolated func processResult(request: VNRequest, error: Error?) -> DepthMapData? {
        if let error {
            print("Depth estimation error: \(error.localizedDescription)")
            return nil
        }

        guard let results = request.results, !results.isEmpty else {
            print("Depth model returned no results")
            return nil
        }

        // Depth model outputs a grayscale image → Vision returns VNPixelBufferObservation
        if let pixelObs = results.first as? VNPixelBufferObservation {
            return extractFromPixelBuffer(pixelObs.pixelBuffer)
        }

        // Fallback: some models return VNCoreMLFeatureValueObservation
        if let featureObs = results.first as? VNCoreMLFeatureValueObservation {
            if let multiArray = featureObs.featureValue.multiArrayValue {
                return extractFromMultiArray(multiArray)
            } else if let imageBuffer = featureObs.featureValue.imageBufferValue {
                return extractFromPixelBuffer(imageBuffer)
            }
        }

        print("Depth output type not recognized: \(type(of: results.first!))")
        return nil
    }

    private nonisolated func extractFromMultiArray(_ array: MLMultiArray) -> DepthMapData? {
        let shape = array.shape.map { $0.intValue }
        let w: Int
        let h: Int
        if shape.count == 3 {
            h = shape[1]; w = shape[2]
        } else if shape.count == 2 {
            h = shape[0]; w = shape[1]
        } else {
            return nil
        }
        return DepthMapData(values: array.toFloatArray(), width: w, height: h)
    }

    private nonisolated func extractFromPixelBuffer(_ buffer: CVPixelBuffer) -> DepthMapData? {
        let w = CVPixelBufferGetWidth(buffer)
        let h = CVPixelBufferGetHeight(buffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(buffer)

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)

        var values = [Float](repeating: 0, count: w * h)

        switch pixelFormat {
        case kCVPixelFormatType_DepthFloat32:
            for y in 0..<h {
                memcpy(&values[y * w], baseAddress.advanced(by: y * bytesPerRow),
                       w * MemoryLayout<Float>.stride)
            }
        case kCVPixelFormatType_OneComponent16Half:
            for y in 0..<h {
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: baseAddress.advanced(by: y * bytesPerRow)),
                    height: 1, width: vImagePixelCount(w), rowBytes: w * 2)
                values.withUnsafeMutableBufferPointer { buf in
                    var dstBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(buf.baseAddress! + y * w),
                        height: 1, width: vImagePixelCount(w), rowBytes: w * 4)
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                }
            }
        case kCVPixelFormatType_OneComponent8:
            for y in 0..<h {
                let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: UInt8.self)
                vDSP_vfltu8(rowPtr, 1, &values[y * w], 1, vDSP_Length(w))
            }
            var divisor: Float = 255.0
            vDSP_vsdiv(values, 1, &divisor, &values, 1, vDSP_Length(w * h))
        default:
            let totalFloats = bytesPerRow * h / MemoryLayout<Float32>.size
            if totalFloats >= w * h {
                for y in 0..<h {
                    memcpy(&values[y * w], baseAddress.advanced(by: y * bytesPerRow),
                           w * MemoryLayout<Float>.stride)
                }
            } else {
                print("Unsupported depth pixel format: \(pixelFormat)")
                return nil
            }
        }

        return DepthMapData(values: values, width: w, height: h)
    }
}
