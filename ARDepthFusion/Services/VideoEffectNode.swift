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

    init(url: URL, naturalSize: CGSize, at position: SCNVector3, scale: Float) {
        super.init()

        // Metal texture cache for zero-copy pixel buffer → GPU texture
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("[VideoEffectNode] Metal not available")
            return
        }
        var cache: CVMetalTextureCache?
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        textureCache = cache

        // Video output: request BGRA so HEVC alpha is pre-composited into one plane
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVPlayerItemVideoOutput(pixelBufferAttributes: outputSettings)
        videoOutput = output

        let playerItem = AVPlayerItem(url: url)
        playerItem.add(output)

        let p = AVPlayer(playerItem: playerItem)
        player = p

        // Loop: seek back to start when video ends
        loopObserver = NotificationCenter.default.addObserver(
            forName: .AVPlayerItemDidPlayToEndTime,
            object: playerItem,
            queue: .main
        ) { [weak p] _ in
            p?.seek(to: .zero)
            p?.play()
        }

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

        // Billboard: always face camera (rotate around Y only)
        let billboard = SCNBillboardConstraint()
        billboard.freeAxes = [.Y]
        self.constraints = [billboard]

        // Display link for per-frame texture updates
        let link = CADisplayLink(target: self, selector: #selector(updateFrame))
        link.add(to: .main, forMode: .common)
        displayLink = link

        p.play()
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

    func stop() {
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
        textureCache = nil
        removeFromParentNode()
    }
}
