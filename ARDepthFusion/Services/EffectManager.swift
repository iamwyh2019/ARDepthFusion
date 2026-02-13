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
    private var metalTextureCache: CVMetalTextureCache?

    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            CVMetalTextureCacheCreate(nil, nil, device, nil, &metalTextureCache)
        }
    }

    deinit {
        // Clean up all pooled players and their observers
        for (_, pool) in preparedPlayers {
            for prepared in pool {
                cleanupPlayer(prepared)
            }
        }
        preparedPlayers.removeAll()
    }

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
                        await self.waitForReadyAndPreroll(prepared.player)
                        var pool = self.preparedPlayers[videoName] ?? []
                        pool.append(prepared)
                        self.preparedPlayers[videoName] = pool
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
        Task {
            await waitForReadyAndPreroll(prepared.player)
            var pool = preparedPlayers[videoName] ?? []
            guard pool.count < poolSize else {
                cleanupPlayer(prepared)
                return
            }
            pool.append(prepared)
            preparedPlayers[videoName] = pool
        }
    }

    func placeEffect(
        type: EffectType,
        objectClass: String,
        at position: SIMD3<Float>,
        scale: Float,
        extent: Object3DExtent? = nil,
        boxDimensions: SIMD3<Float>? = nil,
        cameraYaw: Float? = nil,
        worldPoints: [SIMD3<Float>]? = nil,
        in sceneView: ARSCNView
    ) {
        let scnPosition = SCNVector3(position.x, position.y, position.z)

        let node: SCNNode
        if type == .debugCube {
            let rootNode = SCNNode()
            rootNode.position = scnPosition

            // Rotate around Y to align with OBB principal axis (PCA-aligned or camera-aligned fallback)
            if let yaw = cameraYaw {
                rootNode.eulerAngles.y = yaw
            }

            // Use exact OBB dimensions from caller, fallback to 10cm cube
            let boxW: CGFloat
            let boxH: CGFloat
            let boxD: CGFloat
            if let dims = boxDimensions {
                boxW = CGFloat(dims.x)  // PCA principal axis (or camera-right for fallback)
                boxH = CGFloat(dims.y)  // up axis (gravity)
                boxD = CGFloat(dims.z)  // PCA secondary axis (or camera-forward for fallback)
            } else {
                boxW = 0.1; boxH = 0.1; boxD = 0.1
            }

            let box = SCNBox(width: boxW, height: boxH, length: boxD, chamferRadius: 0)
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.red.withAlphaComponent(0.5)
            material.isDoubleSided = true
            box.materials = [material]
            let cubeNode = SCNNode(geometry: box)
            // Position is already OBB center — no offset needed
            rootNode.addChildNode(cubeNode)
            node = rootNode
        } else if type == .debugMesh {
            guard let points = worldPoints, !points.isEmpty else { return }
            let rootNode = SCNNode()
            rootNode.position = SCNVector3Zero

            let sphere = SCNSphere(radius: 0.005)
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.yellow
            sphere.materials = [material]

            for point in points {
                let dotNode = SCNNode(geometry: sphere)
                dotNode.position = SCNVector3(point.x, point.y, point.z)
                rootNode.addChildNode(dotNode)
            }
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
        // removeFromParentNode() triggers VideoEffectNode.stop() via override
        effect.node.removeFromParentNode()
        placedEffects.removeAll { $0.id == effect.id }
    }

    func clearAll() {
        for effect in placedEffects {
            effect.node.removeFromParentNode()
        }
        placedEffects.removeAll()
        // Reclaim stale Metal texture references
        if let cache = metalTextureCache {
            CVMetalTextureCacheFlush(cache, 0)
        }
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
        Task {
            await waitForReadyAndPreroll(prepared.player)
            var pool = preparedPlayers[videoName] ?? []
            guard pool.count < poolSize else {
                cleanupPlayer(prepared)
                return
            }
            pool.append(prepared)
            preparedPlayers[videoName] = pool
        }
    }

    /// Wait for AVPlayer to reach .readyToPlay, then preroll it.
    /// AVPlayer.preroll crashes if called before status is ready.
    private func waitForReadyAndPreroll(_ player: AVPlayer) async {
        // Wait for readyToPlay status via KVO.
        // Use .initial option to avoid race: if the player becomes ready
        // between our check and observation setup, .initial fires immediately.
        if player.status != .readyToPlay {
            var observation: NSKeyValueObservation?
            var resumed = false
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                observation = player.observe(\.status, options: [.new, .initial]) { player, _ in
                    if !resumed && (player.status == .readyToPlay || player.status == .failed) {
                        resumed = true
                        cont.resume()
                    }
                }
            }
            observation?.invalidate()
        }
        guard player.status == .readyToPlay else { return }
        // Now safe to preroll
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            player.preroll(atRate: 1.0) { _ in
                cont.resume()
            }
        }
    }

    private func cleanupPlayer(_ prepared: PreparedPlayer) {
        NotificationCenter.default.removeObserver(prepared.loopObserver)
        prepared.player.pause()
    }
}
