import SwiftUI
import RealityKit
import ARKit

struct ARContainerView: UIViewRepresentable {
    let coordinator: ARSessionCoordinator
    let arViewRef: (ARView) -> Void

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)

        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = [.horizontal, .vertical]

        // Note: .occlusion is intentionally NOT enabled — it creates a stencil buffer
        // that conflicts with ParticleEmitterComponent's transparent render pass,
        // causing Metal validation assertions. Depth is still available via sceneDepth.
        arView.session.delegate = coordinator
        arView.session.run(config)

        DispatchQueue.main.async { arViewRef(arView) }
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}
