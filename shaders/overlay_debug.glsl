#ifndef OVERLAY_DEBUG_GLSL
#define OVERLAY_DEBUG_GLSL

const int HISTOGRAM_BIN_COUNT = 24;

layout(std140, binding = 0) uniform OverlayUniform
{
    vec4 metrics;
    vec4 histogram[HISTOGRAM_BIN_COUNT];
    vec4 histogramInfo;
} overlay;

vec3 applyOverlay(vec3 color, vec2 fragCoord, vec2 resolution)
{
    return color;
}

#endif // OVERLAY_DEBUG_GLSL
