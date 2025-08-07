/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
 */
#ifndef SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_RATIONAL_HPP
#define SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_RATIONAL_HPP

#include <numeric>
#include <regex>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace hololink::operators {

/**
 * A simple rational number implementation
 * The Rational number is used for setting the signal's sampling interval
 */
struct Rational {
    Rational(const std::string& rational_str)
    { // "numerator/denominator"
        static const std::regex rational_regex(R"((\d+)/(\d+))");
        std::smatch match;
        if (!std::regex_match(rational_str, match, rational_regex)) {
            std::stringstream ss;
            ss << "Unable to parse rational number: " << rational_str;
            throw std::runtime_error(ss.str());
        }
        num_ = std::stoi(match[1]);
        den_ = std::stoi(match[2]);
    }

    Rational(int num = 0, int den = 1)
        : num_(num)
        , den_(den)
    {
    }

    int gcd() const
    {
        return std::gcd(num_, den_);
    }

    template <typename T>
    explicit operator T() const
    {
        return static_cast<T>(num_) / static_cast<T>(den_);
    }

    explicit operator std::string() const
    {
        return std::to_string(num_) + "/" + std::to_string(den_);
    }

    friend std::istream& operator>>(std::istream& is, Rational& rational)
    {
        std::string str;
        is >> str;
        rational = Rational(str);
        return is;
    }

    friend std::ostream& operator<<(std::ostream& os, const Rational& rational)
    {
        os << static_cast<std::string>(rational);
        return os;
    }

    int num_;
    int den_;
};

} // namespace hololink::operators

namespace YAML {

template <>
struct convert<hololink::operators::Rational> {
    static Node encode(const hololink::operators::Rational& rhs)
    {
        Node node;
        node = static_cast<std::string>(rhs);
        return node;
    }

    static bool decode(const Node& node, hololink::operators::Rational& rhs)
    {
        if (!node.IsScalar()) {
            return false;
        }
        rhs = hololink::operators::Rational(node.as<std::string>());
        return true;
    }
};

} // namespace YAML
#endif // SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_RATIONAL_HPP
