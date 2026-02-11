import ARKit
import Combine

final class ARSessionCoordinator: NSObject, ObservableObject, ARSessionDelegate, @unchecked Sendable {
    @Published var hasLiDAR = false
    @Published private(set) var latestFrame: ARFrame?

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let lidarAvailable = frame.sceneDepth != nil
        DispatchQueue.main.async {
            self.latestFrame = frame
            if lidarAvailable && !self.hasLiDAR {
                self.hasLiDAR = true
                print("LiDAR available: true")
            }
        }
    }

    nonisolated func session(_ session: ARSession, didFailWithError error: Error) {
        print("AR session failed: \(error.localizedDescription)")
    }
}
