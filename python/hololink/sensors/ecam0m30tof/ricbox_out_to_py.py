# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import argparse
import sys


def parse_txt(output_file, input_file):
    print("def device_configuration():", file=output_file)
    print("    return [", file=output_file)
    for input_line in input_file:
        s = input_line.strip()
        if s.startswith("#"):
            continue
        if s.startswith("WRITE"):
            o = s.split(", ")
            data = [int(x, 16) for x in o[-1].split(" ")]
            print(
                "        bytes([%s,])," % (", ".join(["0x%02X" % x for x in data]),),
                file=output_file,
            )
            continue
        raise Exception('Unexpected input "%s".' % (s,))
    print("    ]", file=output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spec", help="Filename of .txt file exported from Renesas RICBox"
    )
    parser.add_argument(
        "--output", default=None, help="Filename to write generated code to"
    )
    args = parser.parse_args()

    with open(args.spec, "rt") as input_file:
        if args.output:
            with open(args.output, "wt") as output_file:
                parse_txt(output_file, input_file)
        else:
            parse_txt(sys.stdout, input_file)


if __name__ == "__main__":
    main()
