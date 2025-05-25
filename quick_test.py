#!/usr/bin/env python3
"""Quick test to check MCP setup issue."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import litserve as ls
from litserve.specs.mcp import MCPSpec


class TestAPI(ls.LitAPI):
    def setup(self, device):
        print("TestAPI.setup() called")
        pass

    def decode_request(self, request):
        return request

    def predict(self, x):
        return {"result": f"processed_{x}"}

    def encode_response(self, output):
        return output


def main():
    print("Creating MCP spec...")
    spec = MCPSpec()
    print(f"MCP spec created: {spec}")
    print(f"MCP spec has api attribute: {hasattr(spec, 'api')}")

    print("Creating TestAPI...")
    api = TestAPI()

    print("Creating LitServer...")
    server = ls.LitServer(api, spec=spec)
    print(f"Server created: {server}")

    print("Checking if spec.setup was called...")
    print(f"MCP spec has api attribute after server creation: {hasattr(spec, 'api')}")

    if hasattr(spec, "api"):
        print(f"MCP spec.api: {spec.api}")
    else:
        print("spec.api not set! This is the bug.")
        print("Manually calling spec.setup...")
        spec.setup(server)
        print(f"After manual setup - MCP spec has api: {hasattr(spec, 'api')}")
        if hasattr(spec, "api"):
            print(f"MCP spec.api: {spec.api}")


if __name__ == "__main__":
    main()
