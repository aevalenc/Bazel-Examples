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
        patches = ["//third_party/openmpi:remove_docs_from_subdirectories_to_build.patch"],
    )

openmpi = module_extension(implementation = _openmpi_lib_impl)
