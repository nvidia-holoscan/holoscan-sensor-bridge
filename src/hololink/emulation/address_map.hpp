/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * See README.md for detailed information.
 */

#ifndef ADDRESS_MAP_HPP
#define ADDRESS_MAP_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

/**
 * @brief Sorted array template for address range mapping.
 *
 * Stores key-value pairs where keys are address ranges [start, end) and values
 * are of template type T. Entries are kept sorted by range start; find() uses
 * binary search. No heap/stack used (suitable for bare metal).
 *
 * The primary template uses a fixed-capacity array of size N (no dynamic
 * allocation). A partial specialization for N == 0 (also the default) is
 * provided below that backs storage with std::vector<Entry> and grows on
 * demand; use `AddressMap<T>` to opt in to the dynamic variant.
 *
 * @tparam T The type of values stored in the map
 * @tparam N Fixed capacity, or 0 to select the std::vector-backed specialization
 */
template <typename T, int N = 0>
class AddressMap {
public:
    struct AddressRange {
        uint32_t start;
        uint32_t end;
    };
    struct Entry {
        AddressRange key;
        T value;
    };

    AddressMap()
        : size_(0)
    {
    }

    /**
     * @brief Sorts entries by key.start and checks for overlapping ranges.
     * @return 0 on success, -1 if overlapping ranges detected
     */
    int build()
    {
        if (size_ <= 1) {
            return 0;
        }

        std::sort(data_, data_ + size_, [](const Entry& a, const Entry& b) {
            return a.key.start < b.key.start;
        });

        // Check for overlapping ranges (only valid after sort by start)
        for (int i = 1; i < size_; i++) {
            if (data_[i - 1].key.end > data_[i].key.start) {
                return -1;
            }
        }
        return 0;
    }

    /**
     * @brief Finds index of entry containing addr via binary search.
     * Array must be sorted (build() called). Returns -1 if not found.
     */
    int find(uint32_t addr) const
    {
        int lo = 0;
        int hi = size_ - 1;

        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            const Entry& entry = data_[mid];

            if (addr < entry.key.start) {
                hi = mid - 1;
            } else if (addr >= entry.key.end) {
                lo = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    /**
     * @brief Get a value from the AddressMap by address lookup using range comparison.
     *
     * Searches for an entry where the input address falls within the range
     * [key.start, key.end). Uses binary search on the sorted array.
     *
     * @param addr The address to look up
     * @return Const pointer to the value if found, NULL otherwise
     */
    const T* get(uint32_t addr) const
    {
        int index = find(addr);
        if (index == -1) {
            // fail if the address is not found
            return nullptr;
        }
        return &data_[index].value; // pointer to the value of the entry found
    }

    /**
     * @brief Sets a value by address lookup using range comparison. If the range is not already present, adds a new entry. If the range is already present, updates the value, but the range must match the existing entry exactly.
     * @param key The address range to set the value for. The start address must be less than the end address.
     * @param value The value to set
     * @return 0 on success, -1 if a new element cannot be added, -2 if the range exists and input does not match
     */
    int set(const AddressRange& key, const T& value)
    {
        if (key.start >= key.end) {
            return -1;
        }
        int index = find(key.start);
        if (index < 0) {
            // fail is allocation cannot accommodate the new entry
            if (size_ >= N) {
                return -1;
            }
            // add the new entry
            data_[size_++] = { key, value };
            return 0;
        }
        // fail if the range does not match the existing range
        if (data_[index].key.end != key.end) {
            return -2;
        }
        data_[index].value = value;
        return 0;
    }

private:
    Entry data_[N];
    int size_;
};

/**
 * @brief Partial specialization of AddressMap backed by std::vector.
 *
 * Behaves identically to the fixed-capacity primary template, but storage
 * grows on demand (no fixed upper bound). Selected by writing
 * `AddressMap<T>` (i.e. omitting the capacity argument) or explicitly
 * `AddressMap<T, 0>`. Not suitable for bare-metal targets that forbid heap
 * allocation.
 *
 * @tparam T The type of values stored in the map
 */
template <typename T>
class AddressMap<T, 0> {
public:
    struct AddressRange {
        uint32_t start;
        uint32_t end;
    };
    struct Entry {
        AddressRange key;
        T value;
    };

    AddressMap() = default;

    /**
     * @brief Sorts entries by key.start and checks for overlapping ranges.
     * @return 0 on success, -1 if overlapping ranges detected
     */
    int build()
    {
        if (data_.size() <= 1) {
            return 0;
        }

        std::sort(data_.begin(), data_.end(), [](const Entry& a, const Entry& b) {
            return a.key.start < b.key.start;
        });

        // Check for overlapping ranges (only valid after sort by start)
        for (std::size_t i = 1; i < data_.size(); i++) {
            if (data_[i - 1].key.end > data_[i].key.start) {
                return -1;
            }
        }
        return 0;
    }

    /**
     * @brief Finds index of entry containing addr via binary search.
     * Array must be sorted (build() called). Returns -1 if not found.
     */
    int find(uint32_t addr) const
    {
        int lo = 0;
        int hi = static_cast<int>(data_.size()) - 1;

        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            const Entry& entry = data_[mid];

            if (addr < entry.key.start) {
                hi = mid - 1;
            } else if (addr >= entry.key.end) {
                lo = mid + 1;
            } else {
                return mid;
            }
        }
        return -1;
    }

    /**
     * @brief Get a value from the AddressMap by address lookup using range comparison.
     *
     * Searches for an entry where the input address falls within the range
     * [key.start, key.end). Uses binary search on the sorted array.
     *
     * @param addr The address to look up
     * @return Const pointer to the value if found, NULL otherwise
     */
    const T* get(uint32_t addr) const
    {
        int index = find(addr);
        if (index == -1) {
            // fail if the address is not found
            return nullptr;
        }
        return &data_[index].value; // pointer to the value of the entry found
    }

    /**
     * @brief Sets a value by address lookup using range comparison. If the range is not already present, adds a new entry (growing the underlying vector). If the range is already present, updates the value, but the range must match the existing entry exactly.
     * @param key The address range to set the value for. The start address must be less than the end address.
     * @param value The value to set
     * @return 0 on success, -2 if the range exists and input does not match
     */
    int set(const AddressRange& key, const T& value)
    {
        if (key.start >= key.end) {
            return -1;
        }
        int index = find(key.start);
        if (index < 0) {
            // grow the vector to accommodate the new entry
            data_.push_back({ key, value });
            return 0;
        }
        // fail if the range does not match the existing range
        if (data_[index].key.end != key.end) {
            return -2;
        }
        data_[index].value = value;
        return 0;
    }

private:
    std::vector<Entry> data_;
};

#endif // ADDRESS_MAP_HPP
