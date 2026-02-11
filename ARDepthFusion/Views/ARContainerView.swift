import SwiftUI
import SceneKit
import ARKit

struct ARContainerView: UIViewRepresentable {
    let coordinator: ARSessionCoordinator
    let sceneViewRef: (ARSCNView) -> Void

    func makeUIView(context: Context) -> ARSCNView {
        let sceneView = ARSCNView(frame: .zero)
        sceneView.autoenablesDefaultLighting = true

        let config = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        config.planeDetection = [.horizontal, .vertical]

        sceneView.session.delegate = coordinator
        sceneView.session.run(config)

        DispatchQueue.main.async { sceneViewRef(sceneView) }
        return sceneView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
