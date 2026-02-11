import Foundation

enum DepthFusion {
    static func fuse(relativeDepth: DepthMapData, lidar: LiDARSnapshot, imageWidth: Int, imageHeight: Int) -> DepthFusionResult? {
        let lidarW = lidar.width
        let lidarH = lidar.height

        let daW = relativeDepth.width
        let daH = relativeDepth.height
        let daArray = relativeDepth.values

        // Sample Depth Anything at LiDAR pixel locations
        // DO NOT upsample LiDAR — iterate LiDAR pixels and sample DA via bilinear interp
        var lidarValues: [Float] = []
        var daValues: [Float] = []

        for ly in 0..<lidarH {
            for lx in 0..<lidarW {
                let idx = ly * lidarW + lx
                let confidence = lidar.confidenceValues[idx]
                guard confidence >= 2 else { continue }

                let lidarDepth = lidar.depthValues[idx]
                guard lidarDepth >= 0.1, lidarDepth <= 4.0 else { continue }

                // Map LiDAR pixel to Depth Anything coords
                let daX = Float(lx) / Float(lidarW) * Float(daW)
                let daY = Float(ly) / Float(lidarH) * Float(daH)

                guard let daVal = bilinearSample(
                    daArray, width: daW, height: daH, x: daX, y: daY
                ), daVal > 1e-6 else { continue }

                lidarValues.append(lidarDepth)
                daValues.append(daVal)
            }
        }

        guard lidarValues.count >= 20 else {
            print("DepthFusion: Only \(lidarValues.count) valid pairs (need 20)")
            return nil
        }

        // Least squares: D_metric = alpha * D_relative + beta
        let n = Float(lidarValues.count)
        var sumDA: Float = 0
        var sumLidar: Float = 0
        var sumDA2: Float = 0
        var sumDALidar: Float = 0

        for i in 0..<lidarValues.count {
            let da = daValues[i]
            let lid = lidarValues[i]
            sumDA += da
            sumLidar += lid
            sumDA2 += da * da
            sumDALidar += da * lid
        }

        let denom = n * sumDA2 - sumDA * sumDA
        guard abs(denom) > 1e-10 else {
            print("DepthFusion: Degenerate least squares")
            return nil
        }

        let alpha = (n * sumDALidar - sumDA * sumLidar) / denom
        let beta = (sumLidar - alpha * sumDA) / n

        guard abs(alpha) > 1e-4, abs(alpha) < 1000, beta.isFinite else {
            print("DepthFusion: Degenerate fit alpha=\(alpha) beta=\(beta), skipping")
            return nil
        }

        // Apply to full depth map (clamp to positive range)
        var fusedDepth = [Float](repeating: 0, count: daW * daH)
        for i in 0..<(daW * daH) {
            fusedDepth[i] = max(0.01, alpha * daArray[i] + beta)
        }

        return DepthFusionResult(
            depthMap: fusedDepth,
            width: daW,
            height: daH,
            alpha: alpha,
            beta: beta,
            validPairCount: lidarValues.count,
            imageWidth: imageWidth,
            imageHeight: imageHeight
        )
    }

    private static func bilinearSample(
        _ data: [Float], width: Int, height: Int, x: Float, y: Float
    ) -> Float? {
        let x0 = Int(floor(x))
        let y0 = Int(floor(y))
        let x1 = min(x0 + 1, width - 1)
        let y1 = min(y0 + 1, height - 1)

        guard x0 >= 0, y0 >= 0, x0 < width, y0 < height else { return nil }

        let fx = x - Float(x0)
        let fy = y - Float(y0)

        let d00 = data[y0 * width + x0]
        let d10 = data[y0 * width + x1]
        let d01 = data[y1 * width + x0]
        let d11 = data[y1 * width + x1]

        return d00 * (1 - fx) * (1 - fy)
             + d10 * fx * (1 - fy)
             + d01 * (1 - fx) * fy
             + d11 * fx * fy
    }
}
