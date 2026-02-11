import CoreVideo
import Foundation

struct DepthSample {
    let depth: Float
    let isLiDAR: Bool
}

/// Snapshot of LiDAR depth + confidence data, copied from ARFrame.sceneDepth.
/// This avoids retaining the ARFrame (which starves ARSession's frame pipeline).
struct LiDARSnapshot {
    let depthValues: [Float]
    let confidenceValues: [UInt8]
    let width: Int
    let height: Int

    /// Create by copying pixel buffer data from sceneDepth.
    init?(depthMap: CVPixelBuffer, confidenceMap: CVPixelBuffer?) {
        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }

        guard let depthBase = CVPixelBufferGetBaseAddress(depthMap) else { return nil }
        let depthBytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)

        var depths = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let rowPtr = depthBase.advanced(by: y * depthBytesPerRow)
                .assumingMemoryBound(to: Float32.self)
            for x in 0..<w {
                depths[y * w + x] = rowPtr[x]
            }
        }

        var confidences = [UInt8](repeating: 0, count: w * h)
        if let confMap = confidenceMap {
            CVPixelBufferLockBaseAddress(confMap, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(confMap, .readOnly) }
            if let confBase = CVPixelBufferGetBaseAddress(confMap) {
                let confBytesPerRow = CVPixelBufferGetBytesPerRow(confMap)
                for y in 0..<h {
                    let rowPtr = confBase.advanced(by: y * confBytesPerRow)
                        .assumingMemoryBound(to: UInt8.self)
                    for x in 0..<w {
                        confidences[y * w + x] = rowPtr[x]
                    }
                }
            }
        }

        self.depthValues = depths
        self.confidenceValues = confidences
        self.width = w
        self.height = h
    }
}

struct DepthFusionResult {
    let depthMap: [Float]
    let width: Int
    let height: Int
    let alpha: Float
    let beta: Float
    let validPairCount: Int
    let imageWidth: Int
    let imageHeight: Int

    func sampleDepth(at imagePoint: CGPoint) -> Float? {
        guard width > 0, height > 0 else { return nil }

        // Map image pixel coords to depth map coords
        let scaleX = Float(width) / Float(imageWidth)
        let scaleY = Float(height) / Float(imageHeight)
        let dx = Float(imagePoint.x) * scaleX
        let dy = Float(imagePoint.y) * scaleY

        // Bilinear interpolation
        let x0 = Int(floor(dx))
        let y0 = Int(floor(dy))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)

        guard x0 >= 0, y0 >= 0, x0 < width, y0 < height else { return nil }

        let fx = dx - Float(x0)
        let fy = dy - Float(y0)

        let d00 = depthMap[y0 * width + x0]
        let d10 = depthMap[y0 * width + x1]
        let d01 = depthMap[y1 * width + x0]
        let d11 = depthMap[y1 * width + x1]

        let depth = d00 * (1 - fx) * (1 - fy)
                  + d10 * fx * (1 - fy)
                  + d01 * (1 - fx) * fy
                  + d11 * fx * fy

        guard depth > 0.05, depth < 10.0 else { return nil }
        return depth
    }
}
