"""
Module extension: opencl_headers
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

MPI_VERSION = "5.0.8"

# buildifier: disable=unused-variable
def _openmpi_lib_impl(mctx):
    http_archive(
        name = "openmpi",
        # sha256 = "d4c3f8b1c5e0f2a6b7e8c9d1f2e3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1",
        strip_prefix = "openmpi-" + MPI_VERSION,
        urls = ["https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.8.tar.gz"],
        build_file = "//third_party/openmpi:openmpi.BUILD",
        # patches = ["//third_party/openmpi:add_cpp_header.patch"],
    )

    result = mctx.execute(["pwd"])
    ls_result = mctx.execute(["ls"], working_directory = "/home/se20412/Documents/Personal/Learning/Bazel-Examples/third_party/openmpi")
    print("\n\tPrint working directory: {}".format(result.stdout))  # buildifier: disable=print
    print("List of files: {}".format(ls_result.stdout))  # buildifier: disable=print

    result = result.stdout.split("/")[0:-2]
    output_base = "/".join(result)
    print("\n\tOutputbase: {}".format(output_base))  # buildifier: disable=print

    mctx.file(
        output_base + "/external/+openmpi+openmpi/lol.sh",
        content = """
        #!/bin/bash

        # Configure OpenMPI build
        mkdir -p build/install
        cd build
        if [ $? -ne 0 ]; then
            echo "Failed to create or change to build directory."
            exit 1
        fi

        ../configure --prefix=build/install CC=clang-18 CXX=clang++-18 FC=flang-7
        # Check if the configuration was successful
        if [ $? -ne 0 ]; then
            echo "Configuration failed. Please check the output for errors."
            exit 1
        fi

        """,
        executable = True,
    )

openmpi = module_extension(implementation = _openmpi_lib_impl)
