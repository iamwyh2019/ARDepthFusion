import SceneKit
import ARKit
import Combine

final class EffectManager: ObservableObject {
    @Published var placedEffects: [PlacedEffect] = []

    func placeEffect(
        type: EffectType,
        objectClass: String,
        at position: SIMD3<Float>,
        scale: Float,
        in sceneView: ARSCNView
    ) {
        let rootNode = SCNNode()
        rootNode.position = SCNVector3(position.x, position.y, position.z)

        if type == .debugCube {
            let box = SCNBox(width: 0.1, height: 0.1, length: 0.1, chamferRadius: 0)
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.red
            box.materials = [material]
            let cubeNode = SCNNode(geometry: box)
            rootNode.addChildNode(cubeNode)
        } else if let fileName = type.usdzFileName {
            guard let url = Bundle.main.url(forResource: fileName, withExtension: "usdz") else {
                print("[EffectManager] USDZ file not found: \(fileName).usdz — skipping effect")
                return
            }
            do {
                let usdzScene = try SCNScene(url: url, options: [
                    .checkConsistency: true
                ])
                // Clone all children from the USDZ scene into our root node
                for child in usdzScene.rootNode.childNodes {
                    let clone = child.clone()
                    clone.scale = SCNVector3(scale, scale, scale)
                    rootNode.addChildNode(clone)
                    playAnimations(on: clone)
                }
            } catch {
                print("[EffectManager] Failed to load USDZ \(fileName): \(error.localizedDescription)")
                return
            }
        }

        sceneView.scene.rootNode.addChildNode(rootNode)

        let effect = PlacedEffect(
            type: type,
            objectClass: objectClass,
            node: rootNode
        )
        placedEffects.append(effect)
    }

    func removeEffect(_ effect: PlacedEffect) {
        effect.node.removeFromParentNode()
        placedEffects.removeAll { $0.id == effect.id }
    }

    func clearAll() {
        for effect in placedEffects {
            effect.node.removeFromParentNode()
        }
        placedEffects.removeAll()
    }

    /// Recursively play all animations on a node and its children, looping infinitely.
    private func playAnimations(on node: SCNNode) {
        for key in node.animationKeys {
            if let player = node.animationPlayer(forKey: key) {
                player.animation.repeatCount = .infinity
                player.play()
            }
        }
        for child in node.childNodes {
            playAnimations(on: child)
        }
    }
}
