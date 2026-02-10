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

        arView.environment.sceneUnderstanding.options.insert(.occlusion)
        arView.session.delegate = coordinator
        arView.session.run(config)

        DispatchQueue.main.async { arViewRef(arView) }
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}
