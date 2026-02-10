import ARKit
import RealityKit
import Observation

@Observable
final class ARSessionCoordinator: NSObject, ARSessionDelegate, @unchecked Sendable {
    var hasLiDAR = false
    private(set) var latestFrame: ARFrame?

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let lidarAvailable = frame.sceneDepth != nil
        MainActor.assumeIsolated {
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
