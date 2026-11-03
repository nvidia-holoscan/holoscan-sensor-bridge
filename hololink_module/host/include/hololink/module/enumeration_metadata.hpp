/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ENUMERATION_METADATA_HPP
#define HOLOLINK_MODULE_ENUMERATION_METADATA_HPP

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace hololink::module {

/* EnumerationMetadata crosses the .so boundary by reference. The same
 * toolchain constraint plus the ABI check at module load make
 * sizeof / alignment of this type identical on both sides. */

/**
 * @brief Encapsulates HSB device configuration.
 *
 * Typical EnumerationMetadata lifecycle:
 *
 *  - constructed from received network messages (e.g. BOOTP)
 *  - updated by a loadable module that is specific to that device
 *  - updated by the application layer for specific configuration
 *      changes (e.g. changing some default behavior)
 *  - used to initialize service objects that configure and interface
 *      to HSB functions
 *
 * @code
 *  adapter = hololink_module.Adapter.get_adapter()
 *  # Find a data channel on an HSB device--
 *  # this returns EnumerationMetadata configured with this channel.
 *  hololink_ip = "192.168.0.2"   # as appropriate
 *  discovery_timeout_s = 30
 *  metadata = adapter.wait_for_channel(hololink_ip, discovery_timeout_s)
 *  # That channel is the first data port on a typical
 *  # HsbLite device, which would transfer data from
 *  # sensor 0.  We can change this default and make it
 *  # listen to sensor 1 instead:
 *  sensor_number = 1
 *  adapter.use_sensor(metadata, sensor_number)
 *  # Now instantiate our device driver this way
 *  camera = hololink_module.sensors.vb1940.Vb1940Cam(metadata)
 *  # ...
 * @endcode
 *
 * The framework puts no restrictions on the pairs stored here; a Module and
 * its services are free to coordinate through fields of their own. Values
 * are integers, strings or byte blobs, and every std::map operation is
 * available for working with them.
 */
class EnumerationMetadata
    : public std::map<std::string,
          std::variant<int64_t, std::string, std::vector<uint8_t>>> {
public:
    /**
     * @brief A stored value: an int64_t, a std::string, or a byte blob.
     */
    using Value = mapped_type;

    /**
     * @brief Report whether a field is present, whatever type it holds.
     *
     * @param key Name of the field to test.
     * @return True when the field is present.
     */
    bool contains(const std::string& key) const
    {
        return find(key) != cend();
    }

    /**
     * @brief Fetch the value of a field, which must be present.
     *
     * @tparam T The expected type for the data.
     * @param key Name of the field to fetch.
     * @return The field's value.
     * @throws std::runtime_error if the field is absent or holds another
     * type. Use the overload taking a default value for a field that is
     * genuinely optional.
     */
    template <typename T>
    T get(const std::string& key) const
    {
        const auto it = find(key);
        if (it != cend() && std::holds_alternative<T>(it->second)) {
            return std::get<T>(it->second);
        }
        throw std::runtime_error(
            "While fetching '" + key + "' from enumeration metadata: "
                                       "key is missing or value is of wrong type");
    }

    /* Look up a value of a specific alternative type, falling back to a
     * caller-supplied default when the key is missing.
     *
     * Throws std::runtime_error when the key is present but its variant
     * does not hold T.
     *
     * @tparam T The expected type for the data.
     * @param key Name of the field to fetch.
     * @param default_value Returned when the field is absent.
     * @return The field's value, or `default_value`.
     * @throws std::runtime_error if the field is present but holds another
     * type.
     */
    template <typename T>
    T get(const std::string& key, const T& default_value) const
    {
        const auto it = find(key);
        if (it == cend()) {
            return default_value;
        }
        if (!std::holds_alternative<T>(it->second)) {
            throw std::runtime_error(
                "While fetching '" + key + "' from enumeration metadata: "
                                           "value is of wrong type");
        }
        return std::get<T>(it->second);
    }
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_ENUMERATION_METADATA_HPP
