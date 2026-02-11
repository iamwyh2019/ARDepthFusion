import SceneKit
import ARKit
import AVFoundation
import Combine
import CoreVideo
import Metal

final class EffectManager: ObservableObject {
    @Published var placedEffects: [PlacedEffect] = []

    private var videoCache: [String: (url: URL, naturalSize: CGSize)] = [:]

    // Pool of prerolled players per video type (size 2), auto-replenished on use
    private let poolSize = 2
    private var preparedPlayers: [String: [PreparedPlayer]] = [:]

    // Shared Metal texture cache (one for all VideoEffectNodes)
    private lazy var metalTextureCache: CVMetalTextureCache? = {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        return cache
    }()

    struct PreparedPlayer {
        let player: AVPlayer
        let videoOutput: AVPlayerItemVideoOutput
        let loopObserver: NSObjectProtocol
    }

    /// Phase 1: Preload video asset metadata (track info + naturalSize).
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
                }
            } catch {
                print("[EffectManager] Failed to preload \(videoName): \(error)")
            }
        }
    }

    /// Phase 2: Pre-create and preroll players for each available video effect.
    /// Creates `poolSize` players per type. Call after preloadVideos().
    func prerollAllPlayers() async {
        await withTaskGroup(of: Void.self) { group in
            for (videoName, info) in videoCache {
                for _ in 0..<poolSize {
                    group.addTask { @MainActor in
                        let prepared = self.createPlayer(url: info.url)
                        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                            prepared.player.preroll(atRate: 1.0) { _ in
                                cont.resume()
                            }
                        }
                        var pool = self.preparedPlayers[videoName] ?? []
                        pool.append(prepared)
                        self.preparedPlayers[videoName] = pool
                        print("[EffectManager] Prerolled \(videoName) (\(pool.count)/\(self.poolSize))")
                    }
                }
            }
        }
    }

    /// Prepare a single player on demand. Stores only after preroll completes.
    func preparePlayer(for type: EffectType) {
        guard let videoName = type.videoFileName,
              let info = videoCache[videoName] else { return }
        let currentCount = preparedPlayers[videoName]?.count ?? 0
        guard currentCount < poolSize else { return }

        let prepared = createPlayer(url: info.url)
        prepared.player.preroll(atRate: 1.0) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                var pool = self.preparedPlayers[videoName] ?? []
                guard pool.count < self.poolSize else {
                    self.cleanupPlayer(prepared)
                    return
                }
                pool.append(prepared)
                self.preparedPlayers[videoName] = pool
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
            guard let info = videoCache[videoName],
                  let textureCache = metalTextureCache else {
                print("[EffectManager] Video or Metal not available: \(videoName)")
                return
            }

            // Take a prerolled player from pool, or create on-the-fly as fallback
            let prepared: PreparedPlayer
            if var pool = preparedPlayers[videoName], !pool.isEmpty {
                prepared = pool.removeFirst()
                preparedPlayers[videoName] = pool
            } else {
                prepared = createPlayer(url: info.url)
            }

            let videoNode = VideoEffectNode(
                player: prepared.player,
                videoOutput: prepared.videoOutput,
                loopObserver: prepared.loopObserver,
                textureCache: textureCache,
                naturalSize: info.naturalSize,
                at: scnPosition,
                scale: scale
            )
            node = videoNode

            // Auto-replenish the pool slot in the background
            replenishPlayer(videoName: videoName, url: info.url)
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

    // MARK: - Private

    private func createPlayer(url: URL) -> PreparedPlayer {
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: outputSettings)

        let playerItem = AVPlayerItem(url: url)
        playerItem.add(output)

        let p = AVPlayer(playerItem: playerItem)
        p.isMuted = true

        let observer = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: playerItem,
            queue: .main
        ) { [weak p] _ in
            p?.seek(to: .zero)
            p?.play()
        }

        return PreparedPlayer(player: p, videoOutput: output, loopObserver: observer)
    }

    private func replenishPlayer(videoName: String, url: URL) {
        let currentCount = preparedPlayers[videoName]?.count ?? 0
        guard currentCount < poolSize else { return }

        let prepared = createPlayer(url: url)
        prepared.player.preroll(atRate: 1.0) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self else { return }
                var pool = self.preparedPlayers[videoName] ?? []
                guard pool.count < self.poolSize else {
                    // Pool was filled by another replenish — clean up this one
                    self.cleanupPlayer(prepared)
                    return
                }
                pool.append(prepared)
                self.preparedPlayers[videoName] = pool
            }
        }
    }

    private func cleanupPlayer(_ prepared: PreparedPlayer) {
        NotificationCenter.default.removeObserver(prepared.loopObserver)
        prepared.player.pause()
    }
}
