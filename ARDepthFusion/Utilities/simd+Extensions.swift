import simd
import CoreGraphics

func calculateEffectScale(worldWidth: Float, worldHeight: Float) -> Float {
    let maxDim = max(worldWidth, worldHeight)
    return min(max(maxDim, 0.1), 3.0)
}
