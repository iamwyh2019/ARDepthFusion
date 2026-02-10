import RealityKit
import Foundation

struct PlacedEffect: Identifiable {
    let id = UUID()
    let type: ParticleEffectType
    let objectClass: String
    let anchor: AnchorEntity
}
