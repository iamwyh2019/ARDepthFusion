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
        let fx = intrinsics[0][0]
        let fy = intrinsics[1][1]
        let cx = intrinsics[0][2]
        let cy = intrinsics[1][2]

        // Unproject: pixel -> camera space ray
        let x = (Float(imagePoint.x) - cx) / fx * depth
        let y = (Float(imagePoint.y) - cy) / fy * depth
        let z = -depth  // Camera looks along -Z in ARKit

        let cameraPoint = SIMD4<Float>(x, y, z, 1.0)
        let worldPoint = cameraTransform * cameraPoint

        return SIMD3<Float>(worldPoint.x, worldPoint.y, worldPoint.z)
    }
}
