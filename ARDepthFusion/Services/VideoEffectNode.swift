@preconcurrency import SceneKit
import AVFoundation
import Metal
import CoreVideo

/// A SceneKit node that renders video frames as Metal textures onto a billboard plane.
///
/// IMPORTANT: This class MUST be `nonisolated` because SceneKit accesses SCNNode
/// subclasses from its internal rendering thread. With the project-wide
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor` setting, omitting `nonisolated`
/// would make this class implicitly @MainActor-isolated, causing Timer/CADisplayLink
/// callbacks to silently fail (the `[weak self]` resolves to nil due to actor
/// isolation conflicts with SceneKit's non-main-thread ownership).
nonisolated class VideoEffectNode: SCNNode, @unchecked Sendable {
    private var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var updateTimer: Timer?
    private var textureCache: CVMetalTextureCache?
    private var currentTexture: CVMetalTexture?
    private var loopObserver: NSObjectProtocol?
    private var stopped = false

    /// Create with pre-prepared player components and a shared texture cache.
    init(player: AVPlayer,
         videoOutput: AVPlayerItemVideoOutput,
         loopObserver: NSObjectProtocol,
         textureCache: CVMetalTextureCache,
         naturalSize: CGSize,
         at position: SCNVector3,
         scale: Float) {
        super.init()

        self.player = player
        self.videoOutput = videoOutput
        self.loopObserver = loopObserver
        self.textureCache = textureCache

        let aspectRatio = naturalSize.width / naturalSize.height

        // Plane geometry sized by scale (2x multiplier)
        let width = CGFloat(scale) * 2.0
        let height = width / aspectRatio
        let plane = SCNPlane(width: width, height: height)

        let material = SCNMaterial()
        material.diffuse.contents = UIColor.clear
        material.isDoubleSided = true
        material.blendMode = .alpha
        material.transparencyMode = .aOne
        material.writesToDepthBuffer = true
        material.readsFromDepthBuffer = true
        plane.materials = [material]

        self.geometry = plane
        self.position = position

        // Billboard: rotate around Y axis only so the plane always faces the
        // camera horizontally but stays upright (no tilt). This keeps effects
        // like fire and explosions looking natural. Use [.X, .Y] to fully
        // face the camera from any angle.
        let billboard = SCNBillboardConstraint()
        billboard.freeAxes = [.Y]
        self.constraints = [billboard]

        // Schedule timer on the main thread explicitly.
        // Block-based Timer avoids @MainActor + @objc selector dispatch issues
        // that break CADisplayLink on newer Swift runtimes.
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            let timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
                self?.updateFrame()
            }
            RunLoop.main.add(timer, forMode: .common)
            self.updateTimer = timer
        }

        // Player was prerolled — start immediately
        player.play()
    }

    private func updateFrame() {
        guard let output = videoOutput,
              let currentItem = player?.currentItem,
              let textureCache else { return }

        let time = currentItem.currentTime()
        guard output.hasNewPixelBuffer(forItemTime: time),
              let pixelBuffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else {
            return
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, pixelBuffer, nil,
            .bgra8Unorm, width, height, 0, &cvTexture
        )

        guard status == kCVReturnSuccess,
              let cvTexture,
              let metalTexture = CVMetalTextureGetTexture(cvTexture) else { return }

        currentTexture = cvTexture  // Keep alive while MTLTexture is in use
        geometry?.firstMaterial?.diffuse.contents = metalTexture
    }

    required init?(coder: NSCoder) { fatalError() }

    /// When removed from scene, clean up resources.
    /// Only call stop() if the node is actually in a scene (has a parent).
    /// SceneKit's addChildNode() may internally call removeFromParentNode()
    /// on the node before attaching it, which would incorrectly trigger stop().
    override func removeFromParentNode() {
        if parent != nil {
            stop()
        }
        super.removeFromParentNode()
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        updateTimer?.invalidate()
        updateTimer = nil
        if let observer = loopObserver {
            NotificationCenter.default.removeObserver(observer)
            loopObserver = nil
        }
        player?.pause()
        player = nil
        videoOutput = nil
        currentTexture = nil
        // textureCache is shared — don't nil it, EffectManager owns it
    }

    deinit {
        stop()
    }
}
