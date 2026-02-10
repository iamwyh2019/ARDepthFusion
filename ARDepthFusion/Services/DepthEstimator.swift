import Accelerate
import CoreML
import CoreImage
import CoreVideo

struct DepthMapData: Sendable {
    let values: [Float]
    let width: Int
    let height: Int
}

final class DepthEstimator: @unchecked Sendable {
    static let shared = DepthEstimator()

    private var model: MLModel?

    private init() {}

    private func loadModel() throws -> MLModel {
        if let model { return model }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        let model = try DepthAnythingV2SmallF16(configuration: config).model
        self.model = model
        print("Depth Anything V2 model loaded")
        return model
    }

    nonisolated func estimateDepth(pixelBuffer: CVPixelBuffer) async -> DepthMapData? {
        let start = CFAbsoluteTimeGetCurrent()

        do {
            let model = try await MainActor.run { try loadModel() }
            let input = try MLDictionaryFeatureProvider(
                dictionary: ["image": pixelBuffer]
            )
            let output = try await Task.detached(priority: .userInitiated) {
                try model.prediction(from: input)
            }.value

            guard let depthFeature = output.featureValue(for: "depth") else {
                print("Depth model returned no depth output")
                return nil
            }

            let depthData: DepthMapData?

            if let multiArray = depthFeature.multiArrayValue {
                depthData = extractFromMultiArray(multiArray)
            } else if let imageBuffer = depthFeature.imageBufferValue {
                depthData = extractFromPixelBuffer(imageBuffer)
            } else {
                print("Depth output is neither multiArray nor image")
                return nil
            }

            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            print(String(format: "Depth Anything inference: %.0fms", elapsed))
            return depthData
        } catch {
            print("Depth estimation failed: \(error.localizedDescription)")
            return nil
        }
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
            // Float32 grayscale
            for y in 0..<h {
                let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: Float32.self)
                for x in 0..<w {
                    values[y * w + x] = rowPtr[x]
                }
            }
        case kCVPixelFormatType_OneComponent16Half:
            // Float16 grayscale
            for y in 0..<h {
                let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: UInt16.self)
                for x in 0..<w {
                    values[y * w + x] = float16ToFloat32(rowPtr[x])
                }
            }
        case kCVPixelFormatType_OneComponent8:
            // UInt8 grayscale, normalize to 0-1
            for y in 0..<h {
                let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
                    .assumingMemoryBound(to: UInt8.self)
                for x in 0..<w {
                    values[y * w + x] = Float(rowPtr[x]) / 255.0
                }
            }
        default:
            // Try treating as float32 single channel
            let totalFloats = bytesPerRow * h / MemoryLayout<Float32>.size
            if totalFloats >= w * h {
                for y in 0..<h {
                    let rowPtr = baseAddress.advanced(by: y * bytesPerRow)
                        .assumingMemoryBound(to: Float32.self)
                    for x in 0..<w {
                        values[y * w + x] = rowPtr[x]
                    }
                }
            } else {
                print("Unsupported depth pixel format: \(pixelFormat)")
                return nil
            }
        }

        return DepthMapData(values: values, width: w, height: h)
    }

    private nonisolated func float16ToFloat32(_ h: UInt16) -> Float {
        var f16 = h
        var f32: Float = 0
        withUnsafePointer(to: &f16) { src in
            withUnsafeMutablePointer(to: &f32) { dst in
                var srcBuf = vImage_Buffer(
                    data: UnsafeMutablePointer(mutating: src),
                    height: 1, width: 1, rowBytes: 2
                )
                var dstBuf = vImage_Buffer(
                    data: dst, height: 1, width: 1, rowBytes: 4
                )
                vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
            }
        }
        return f32
    }
}
