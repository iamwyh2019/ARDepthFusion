import SwiftUI

struct ControlPanelView: View {
    let statusText: String
    let isDetecting: Bool
    let detectionCount: Int
    let effectCount: Int
    let onDetect: () -> Void
    let onClearAll: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Text(statusText)
                .font(.caption)
                .foregroundColor(.secondary)

            HStack(spacing: 16) {
                Button(action: onDetect) {
                    HStack {
                        if isDetecting {
                            ProgressView()
                                .tint(.white)
                        }
                        Text(isDetecting ? "Detecting..." : "Detect")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(isDetecting ? Color.gray : Color.blue)
                    .foregroundColor(.white)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .disabled(isDetecting)

                if effectCount > 0 {
                    Button(action: onClearAll) {
                        Text("Clear All")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 12)
                            .background(Color.red.opacity(0.8))
                            .foregroundColor(.white)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }
                }
            }
        }
        .padding()
        .background(.ultraThinMaterial)
    }
}
