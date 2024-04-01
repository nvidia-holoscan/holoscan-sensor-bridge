# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import datetime
import hashlib
import tempfile
import yaml
import zipfile

import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--org",
        default="nvidia",
        help="NGC 'org' with project files.",
    )
    parser.add_argument(
        "--team",
        default="clara-holoscan",
        help="NGC 'team' with project files.",
    )
    parser.add_argument(
        "--project",
        default="holoscan_sensor_bridge_fpga_ip",
        help="NGC resource with project files.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Componoent version (e.g. \"2402\")",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.yaml",
        help="Manifest file to write with programming data.",
    )
    args = parser.parse_args()
    # ...
    org = args.org
    team = args.team
    project = args.project
    version = args.version
    utc = datetime.timezone.utc
    now = datetime.datetime.now(utc)
    #
    zip_url = f"https://api.ngc.nvidia.com/v2/resources/{org}/{team}/{project}/versions/{version}/zip"
    zip_request = requests.get(
        zip_url,
        headers={
            "Content-Type": "application/json",
        },
    )
    if zip_request.status_code != requests.codes.ok:
        raise Exception(f"Unable to fetch \"{zip_url}\"; status={zip_request.status_code}")
    #
    md5 = hashlib.md5(zip_request.content)
    hololink = {
        "archive": {
            "type": "zip",
            "url": zip_url,
            "version": version,
            "md5": md5.hexdigest(),
            "enrollment_date": now.isoformat(),
        },
    }
    hololink_images = { }
    others = [ ]
    with tempfile.TemporaryFile() as f:
        # Save the zip file content
        f.write(zip_request.content)
        # Now go back and read it as a zip file
        f.seek(0)
        zip_file = zipfile.ZipFile(f)
        for name in zip_file.namelist():
            content = zip_file.read(name)
            md5 = hashlib.md5(content)
            print(f"{name=} {len(content)=} {md5.hexdigest()=}")
            image = {
                "content": name,
                "size": len(content),
                "md5": md5.hexdigest(),
            }
            if "cpnx" in name:
                assert hololink_images.get("cpnx") is None
                hololink_images["cpnx"] = image
            elif "clnx" in name:
                assert hololink_images.get("clnx") is None
                hololink_images["clnx"] = image
            elif "LICENSE" in name.upper():
                licenses = hololink.setdefault("licenses", [])
                licenses.append({"name": name})
            else:
                hololink.setdefault("other_content", []).append(image)
    assert len(hololink_images) > 0
    hololink["images"] = hololink_images
    # Write the metadata to the manifest file
    manifest = {
        "hololink": hololink,
    }
    with open(args.manifest, "wt") as f:
        f.write(yaml.dump(manifest, default_flow_style=False))

if __name__ == "__main__":
    main()
