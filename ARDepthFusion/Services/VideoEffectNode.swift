import SceneKit
import AVFoundation
import Metal
import CoreVideo

class VideoEffectNode: SCNNode {
    private var player: AVPlayer?
    private var videoOutput: AVPlayerItemVideoOutput?
    private var displayLink: CADisplayLink?
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

        // Display link for per-frame texture updates
        let link = CADisplayLink(target: self, selector: #selector(updateFrame))
        link.add(to: .main, forMode: .common)
        displayLink = link

        // Player was prerolled — start immediately
        player.play()
    }

    @objc private func updateFrame() {
        guard let output = videoOutput,
              let currentItem = player?.currentItem,
              let textureCache else { return }

        let time = currentItem.currentTime()
        guard output.hasNewPixelBuffer(forItemTime: time),
              let pixelBuffer = output.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil) else { return }

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

    /// Safety net: if SceneKit removes the node (e.g. session reset), ensure
    /// the CADisplayLink is invalidated to break the retain cycle.
    override func removeFromParentNode() {
        stop()
        super.removeFromParentNode()
    }

    func stop() {
        guard !stopped else { return }
        stopped = true
        displayLink?.invalidate()
        displayLink = nil
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
}
