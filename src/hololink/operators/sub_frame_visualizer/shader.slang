// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fullscreen triangle sampling the exported frame image. The scissor rect set per band clips the
// draw to that band's rows while the UVs stay full-frame, so each band lands at its own position.
Texture2D frameTexture;
SamplerState frameSampler;

// Per-draw data; the layout must match ShaderPushConstants in sub_frame_visualizer.cpp.
struct PushConstants {
    uint flags;
    // Vertical sampling bounds (texel centres of the rows this draw may read).
    float uv_min_y;
    float uv_max_y;
};
[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;
static const uint kFlagGrayscaleReplicate = 1u;

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 uv_coord : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID) {
    float2 position = float2(vertex_id == 1 ? 3.0 : -1.0, vertex_id == 2 ? 3.0 : -1.0);
    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    output.uv_coord = position * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_TARGET {
    // The scissor bounds the destination rows, not the filter footprint: clamp the source row
    // so linear filtering at a band edge cannot read a neighbouring row that has not been
    // uploaded for this frame yet.
    float2 uv = input.uv_coord;
    uv.y = clamp(uv.y, pc.uv_min_y, pc.uv_max_y);
    float4 texel = frameTexture.Sample(frameSampler, uv);
    if ((pc.flags & kFlagGrayscaleReplicate) != 0u) {
        // Single-channel (R16) source: replicate red to gray, opaque.
        texel = float4(texel.rrr, 1.0);
    }
    return texel;
}
