import CoreML
import Foundation

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
