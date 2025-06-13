#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>

namespace rs2::utils {

struct float3 {
    float x, y, z;
    float3 operator*(float t) const { return {x * t, y * t, z * t}; }
    float3 operator+(const float3& o) const { return {x + o.x, y + o.y, z + o.z}; }
};

// Interpolated color map
class ColorMap {
public:
    ColorMap(const std::vector<float3>& values, int steps = 4000) {
        const float step = 1.0f / (values.size() - 1);
        for (size_t i = 0; i < values.size(); ++i)
            _map[i * step] = values[i];
        initialize(steps);
    }

    float3 get(float value) const {
        float t = std::clamp((value - _min) / (_max - _min), 0.f, 1.f);
        return _cache[static_cast<size_t>(t * (_cache.size() - 1))];
    }

private:
    void initialize(int steps) {
        _min = _map.begin()->first;
        _max = _map.rbegin()->first;

        _cache.resize(steps + 1);
        for (int i = 0; i <= steps; ++i) {
            float t = static_cast<float>(i) / steps;
            float key = _min + t * (_max - _min);
            _cache[i] = interpolate(key);
        }
    }

    float3 interpolate(float v) const {
        auto it = _map.lower_bound(v);
        if (it == _map.begin()) return it->second;
        if (it == _map.end()) return std::prev(it)->second;

        auto hi = it;
        auto lo = std::prev(it);
        float t = (v - lo->first) / (hi->first - lo->first);
        return lo->second * (1.f - t) + hi->second * t;
    }

    std::map<float, float3> _map;
    std::vector<float3> _cache;
    float _min, _max;
};

// Colorizer for depth frames
class Colorizer {
public:
    Colorizer()
        : _colormap({
              {0.f, 0.f, 255.f},   // Blue
              {0.f, 255.f, 255.f}, // Cyan
              {255.f, 255.f, 0.f}, // Yellow
              {255.f, 0.f, 0.f},   // Red
              {50.f, 0.f, 0.f}     // Dark Red
          }),
          _depth_units(0.001f),
          _equalize(true),
          _min(0.3f),
          _max(4.0f) {}

    void colorize(const uint16_t* depth_data, uint8_t* rgb_data, int width, int height) {
        const size_t size = width * height;
        std::vector<int> hist(0x10000, 0);

        if (_equalize) {
            for (size_t i = 0; i < size; ++i)
                if (depth_data[i] > 0) hist[depth_data[i]]++;
            for (size_t i = 1; i < hist.size(); ++i)
                hist[i] += hist[i - 1];
        }

        const int total = hist[0xFFFF];

        for (size_t i = 0; i < size; ++i) {
            uint16_t d = depth_data[i];

            // Black pixel for depth = 0
            if (d == 0) {
                rgb_data[3 * i + 0] = 0;
                rgb_data[3 * i + 1] = 0;
                rgb_data[3 * i + 2] = 0;
                continue;
            }

            float norm = 0.f;
            if (_equalize) {
                norm = (total > 0) ? static_cast<float>(hist[d]) / total : 0.f;
            } else {
                float depth_m = d * _depth_units;
                norm = std::clamp((depth_m - _min) / (_max - _min), 0.f, 1.f);
            }

            auto c = _colormap.get(norm);
            rgb_data[3 * i + 0] = static_cast<uint8_t>(c.x);
            rgb_data[3 * i + 1] = static_cast<uint8_t>(c.y);
            rgb_data[3 * i + 2] = static_cast<uint8_t>(c.z);
        }
    }

    void set_equalize(bool e) { _equalize = e; }
    void set_range(float min_m, float max_m) { _min = min_m; _max = max_m; }
    void set_depth_units(float du) { _depth_units = du; }

private:
    float _depth_units;
    float _min, _max;
    bool _equalize;
    ColorMap _colormap;
};

}  // namespace realsense::utils
