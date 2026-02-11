import SwiftUI

struct EffectListView: View {
    @ObservedObject var effectManager: EffectManager

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
                            effectManager.removeEffect(effect)
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
