import SceneKit
import Foundation

struct PlacedEffect: Identifiable {
    let id = UUID()
    let type: EffectType
    let objectClass: String
    let node: SCNNode
}
