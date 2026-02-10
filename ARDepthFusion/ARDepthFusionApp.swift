//
//  ARDepthFusionApp.swift
//  ARDepthFusion
//
//  Created by 吴宇恒 on 2/10/26.
//

import SwiftUI

@main
struct ARDepthFusionApp: App {
    init() {
        // Load both ML models at launch so first detection is fast
        ObjectDetectionService.shared.initialize()
        _ = DepthEstimator.shared
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
