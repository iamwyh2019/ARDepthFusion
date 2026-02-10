import simd
import CoreGraphics

func calculateEffectScale(
    boundingBox: CGRect,
    depth: Float,
    intrinsics: simd_float3x3,
    imageWidth: Float
) -> Float {
    let fx = intrinsics[0][0]
    let bboxWidthPixels = Float(boundingBox.width)

    // Convert pixel width to real-world meters at the given depth
    let worldWidth = bboxWidthPixels * depth / fx

    // Clamp to reasonable effect size
    return min(max(worldWidth, 0.1), 3.0)
}
