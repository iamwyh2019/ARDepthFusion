import SceneKit
import ARKit
import AVFoundation
import Combine

final class EffectManager: ObservableObject {
    @Published var placedEffects: [PlacedEffect] = []

    private var videoCache: [String: (url: URL, naturalSize: CGSize)] = [:]

    /// Preload all video assets at app launch (reads track metadata + naturalSize).
    func preloadVideos() async {
        for type in EffectType.allCases {
            guard let videoName = type.videoFileName,
                  let url = Bundle.main.url(forResource: videoName, withExtension: "mov") else { continue }

            let asset = AVAsset(url: url)
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                if let track = tracks.first {
                    let size = try await track.load(.naturalSize)
                    videoCache[videoName] = (url: url, naturalSize: size)
                    print("[EffectManager] Preloaded \(videoName): \(size)")
                }
            } catch {
                print("[EffectManager] Failed to preload \(videoName): \(error)")
            }
        }
    }

    func placeEffect(
        type: EffectType,
        objectClass: String,
        at position: SIMD3<Float>,
        scale: Float,
        in sceneView: ARSCNView
    ) {
        let scnPosition = SCNVector3(position.x, position.y, position.z)

        let node: SCNNode
        if type == .debugCube {
            let rootNode = SCNNode()
            rootNode.position = scnPosition
            let box = SCNBox(width: 0.1, height: 0.1, length: 0.1, chamferRadius: 0)
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.red
            box.materials = [material]
            let cubeNode = SCNNode(geometry: box)
            rootNode.addChildNode(cubeNode)
            node = rootNode
        } else if let videoName = type.videoFileName {
            let cached = videoCache[videoName]
            guard let url = cached?.url ?? Bundle.main.url(forResource: videoName, withExtension: "mov") else {
                print("[EffectManager] Video not found: \(videoName).mov")
                return
            }
            let naturalSize = cached?.naturalSize ?? CGSize(width: 1920, height: 1080)
            let videoNode = VideoEffectNode(url: url, naturalSize: naturalSize, at: scnPosition, scale: scale)
            node = videoNode
        } else {
            return
        }

        sceneView.scene.rootNode.addChildNode(node)

        let effect = PlacedEffect(
            type: type,
            objectClass: objectClass,
            node: node
        )
        placedEffects.append(effect)
    }

    func removeEffect(_ effect: PlacedEffect) {
        if let videoNode = effect.node as? VideoEffectNode {
            videoNode.stop()
        } else {
            effect.node.removeFromParentNode()
        }
        placedEffects.removeAll { $0.id == effect.id }
    }

    func clearAll() {
        for effect in placedEffects {
            if let videoNode = effect.node as? VideoEffectNode {
                videoNode.stop()
            } else {
                effect.node.removeFromParentNode()
            }
        }
        placedEffects.removeAll()
    }
}
