# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#
# Create a manifest.yaml that trains the local downloader to
# program just a single FPGA with a local bit file.  Users
# are trusted to ensure that the bit file is correct for the
# device given here.
#

import argparse
import datetime
import hashlib
import os
import requests
import urllib.parse
import yaml

def measure(metadata, content):
    md5 = hashlib.md5(content)
    metadata.update({
        "size": len(content),
        "md5": md5.hexdigest(),
    })

def fetch_url(url):
    # Given a url, extract just the filename
    p = urllib.parse.urlparse(url)
    image = p.path.split("/")[-1]
    # Fetch the content
    request = requests.get(
        url,
        headers={
            "Content-Type": "binary/octet-stream",
        },
    )
    if request.status_code != requests.codes.ok:
        raise Exception(
            f'Unable to fetch "{url}"; status={request.status_code}'
        )
    content = request.content
    # build a metadata
    metadata = {
        "url": url,
    }
    return image, metadata, content

def fetch_file(filename):
    p = os.path.split(filename)
    image = p[-1]
    with open(filename, "rb") as f:
        content = f.read()
    metadata = {
        "filename": filename,
    }
    return image, metadata, content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        required=True,
        help="Component version (e.g. \"2402\")",
    )
    parser.add_argument(
        "--manifest",
        help="Manifest file to write with programming data.",
    )
    parser.add_argument(
        "--cpnx-file",
        help="CPNX bit file to program, fetched from a local copy.",
    )
    parser.add_argument(
        "--clnx-file",
        help="CLNX bit file to program, fetched from a local copy.",
    )
    parser.add_argument(
        "--stratix-file",
        help="Stratix-10 rpd file to program, fetched from a local copy.",
    )
    parser.add_argument(
        "--eula-file",
        help="EULA, fetched from a local file.",
    )
    parser.add_argument(
        "--cpnx-url",
        help="CPNX bit file to program, fetched from a URL.",
    )
    parser.add_argument(
        "--clnx-url",
        help="CLNX bit file to program, fetched from a URL.",
    )
    parser.add_argument(
        "--stratix-url",
        help="Stratix-10 rpd file to program, fetched from a URL.",
    )
    parser.add_argument(
        "--eula-url",
        help="EULA, fetched from a URL",
    )
    parser.add_argument(
        "--strategy",
        help="Specify the strategy to use with this manifest.",
    )
    parser.add_argument(
        "--fpga-uuid",
        action="append",
        help="What FPGA UUID is appropriate for this configuration.",
    )
    args = parser.parse_args()
    # ...
    version = args.version
    utc = datetime.timezone.utc
    now = datetime.datetime.now(utc)
    cpnx_file = args.cpnx_file
    clnx_file = args.clnx_file
    stratix_file = args.stratix_file
    eula_file = args.eula_file
    cpnx_url = args.cpnx_url
    clnx_url = args.clnx_url
    stratix_url = args.stratix_url
    eula_url = args.eula_url
    fpga_uuid = args.fpga_uuid
    if (fpga_uuid is None) or (len(fpga_uuid) < 1):
        parser.error("At least one --fpga-uuid must be specified.")
    if not any([cpnx_file, clnx_file, stratix_file, cpnx_url, clnx_url, stratix_url]):
        parser.error("At least one of --cpnx-file, --clnx-file, --stratix-file, --cpnx-url, --clnx-url, or --stratix-url must be specified.")
    strategy = args.strategy
    if strategy is None:
        strategy = "sensor_bridge_10"
        if (stratix_file is not None) or (stratix_url is not None):
            strategy = "sensor_bridge_100"
    # We should never fail this due to the parser.error check above.
    if strategy is None:
        parser.error("Unable to compute strategy for this configuration.")
    #
    hololink = {
        "archive": {
            "version": version,
            "enrollment_date": now.isoformat(),
        },
        "content": {
        },
        "strategy": strategy,
    }
    images = [ ]
    def measure_file(filename, context):
        image, metadata, content = fetch_file(filename)
        measure(metadata, content)
        nonlocal hololink
        hololink["content"][image] = metadata
        nonlocal images
        images.append({
            "content": image,
            "context": context,
        })
    def measure_url(url, context):
        image, metadata, content = fetch_url(url)
        measure(metadata, content)
        nonlocal hololink
        hololink["content"][image] = metadata
        nonlocal images
        images.append({
            "content": image,
            "context": context,
        })
    if cpnx_file is not None:
        measure_file(cpnx_file, "cpnx")
    if clnx_file is not None:
        measure_file(clnx_file, "clnx")
    if stratix_file is not None:
        measure_file(stratix_file, "stratix")
    if eula_file is not None:
        image, metadata, content = fetch_file(eula_file)
        measure(metadata, content)
        hololink["content"][image] = metadata
        licenses = hololink.setdefault("licenses", [])
        licenses.append(image)
    if cpnx_url is not None:
        measure_url(cpnx_url, "cpnx")
    if clnx_url is not None:
        measure_url(clnx_url, "clnx")
    if stratix_url is not None:
        measure_url(stratix_url, "stratix")
    if eula_url is not None:
        image, metadata, content = fetch_url(eula_url)
        measure(metadata, content)
        hololink["content"][image] = metadata
        licenses = hololink.setdefault("licenses", [])
        licenses.append(image)

    hololink["images"] = images
    hololink["fpga_uuid"] = fpga_uuid
    # Write the metadata to the manifest file
    manifest = {
        "hololink": hololink,
    }
    with open(args.manifest, "wt") as f:
        f.write(yaml.dump(manifest, default_flow_style=False))

if __name__ == "__main__":
    main()
