import SwiftUI
import RealityKit

struct EffectListView: View {
    var effectManager: EffectManager
    let arView: ARView?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                ForEach(effectManager.placedEffects) { effect in
                    HStack(spacing: 6) {
                        Image(systemName: effect.type.icon)
                            .font(.caption)
                        Text(effect.objectClass)
                            .font(.caption2)
                            .lineLimit(1)
                        Button {
                            if let arView {
                                effectManager.removeEffect(effect, from: arView)
                            }
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial, in: Capsule())
                }
            }
            .padding(.horizontal)
        }
        .padding(.bottom, 4)
    }
}
