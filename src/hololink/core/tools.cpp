/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <dirent.h>

#include "tools.hpp"

static const char* infiniband_device_directory = "/sys/class/infiniband";

namespace hololink::core {

class Directory {
public:
    class Entry {
    public:
        Entry(::dirent* entry = nullptr)
            : entry_(entry)
        {
        }

        operator bool() const
        {
            return static_cast<bool>(entry_);
        }

        std::string_view name() const
        {
            return std::string_view(entry_->d_name);
        }

        friend bool operator<(const Entry& lhs, const Entry& rhs)
        {
            return lhs.name() < rhs.name();
        }

    private:
        ::dirent* entry_;
    };
    class Iterator {
    public:
        using value_type = Entry;
        using pointer = value_type*;
        using reference = value_type&;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;

        // Constructors / Destructor
        Iterator() = default;
        Iterator(const std::string& dir_path)
            : dir_(::opendir(dir_path.c_str()))
        {
            if (!dir_) {
                std::stringstream ss;
                ss << "Unable to open directory '" << dir_path << "'";
                throw std::runtime_error(ss.str());
            }
            ++(*this);
        }
        Iterator(const Iterator&) = delete;
        Iterator(Iterator&& other)
            : Iterator()
        {
            swap(*this, other);
        }
        Iterator& operator=(const Iterator&) = delete;
        Iterator& operator=(Iterator&& other)
        {
            swap(*this, other);
            return *this;
        };
        ~Iterator()
        {
            if (dir_)
                ::closedir(dir_);
        }

        reference operator*()
        {
            return entry_;
        }

        pointer operator->()
        {
            return &entry_;
        }

        // Increment operator
        Iterator& operator++()
        {
            while ((entry_ = readdir(dir_)) && (entry_.name() == "." || entry_.name() == ".."))
                ;
            return *this;
        }

        // Equality/Inequality operators
        bool operator==(const Iterator& other) const
        {
            return entry_ == other.entry_;
        }
        bool operator!=(const Iterator& other) const
        {
            return !(*this == other);
        }

        friend void swap(Iterator& lhs, Iterator& rhs) noexcept
        {
            using std::swap;
            swap(lhs.dir_, rhs.dir_);
            swap(lhs.entry_, rhs.entry_);
        }

    private:
        ::DIR* dir_ = nullptr;
        Entry entry_;
    };

    Directory(const std::string& dir_path)
        : dir_path_(dir_path)
    {
    }

    Iterator begin() const
    {
        return Iterator(dir_path_);
    }

    Iterator end() const
    {
        return Iterator();
    }

private:
    std::string dir_path_;
};

std::vector<std::string> infiniband_devices()
{
    try {
        Directory directory(infiniband_device_directory);
        std::vector<Directory::Entry> directory_entries;
        for (auto iter = directory.begin(); iter != directory.end(); ++iter)
            directory_entries.push_back(*iter);
        std::sort(directory_entries.begin(), directory_entries.end());

        std::vector<std::string> devices;
        std::transform(directory_entries.begin(), directory_entries.end(), std::back_inserter(devices), [](const Directory::Entry& entry) {
            return std::string(entry.name());
        });
        return devices;
    } catch (const std::exception& e) {
        return std::vector<std::string>();
    }
}

} // namespace hololink::core