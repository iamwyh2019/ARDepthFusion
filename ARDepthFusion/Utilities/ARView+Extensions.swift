import ARKit
import RealityKit
import simd

extension ARView {
    func worldPosition(
        imagePoint: CGPoint,
        depth: Float,
        frame: ARFrame
    ) -> SIMD3<Float>? {
        let intrinsics = frame.camera.intrinsics
        let cameraTransform = frame.camera.transform

        // Image point is in camera image space (landscape-right, 1920x1440)
        // simd_float3x3 is column-major: matrix[column][row]
        let fx = intrinsics[0][0]  // column 0, row 0
        let fy = intrinsics[1][1]  // column 1, row 1
        let cx = intrinsics[2][0]  // column 2, row 0 (principal point x)
        let cy = intrinsics[2][1]  // column 2, row 1 (principal point y)

        // Unproject: pixel -> camera space ray
        let x = (Float(imagePoint.x) - cx) / fx * depth
        let y = (Float(imagePoint.y) - cy) / fy * depth
        let z = -depth  // Camera looks along -Z in ARKit

        let cameraPoint = SIMD4<Float>(x, y, z, 1.0)
        let worldPoint = cameraTransform * cameraPoint

        return SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
    }
}
