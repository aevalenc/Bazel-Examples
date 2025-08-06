load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

filegroup(
    name = "openmpi_srcs",
    srcs = glob(["**/*"]),
)

configure_make(
    name = "configure_mpi_clang18",
    args = ["-j4"],
    # configure_in_place = True,
    configure_options = select(
        {
            "//conditions:default": [
                "CC=/usr/bin/clang-18",
                "CXX=/usr/bin/clang++-18",
            ],
            "@platforms//os:macos": [
                # "--with-libevent-libdir=/opt/homebrew/Cellar/libevent/2.1.12_1",
                # "--with-hwloc-libdir=/opt/homebrew/Cellar/hwloc/2.12.1",
                "--enable-mpi-fortran=no",
                "--with-hwloc=internal",
                "--with-libevent=internal",
                "CC=/opt/homebrew/opt/llvm@18/bin/clang",
                "CXX=/opt/homebrew/opt/llvm@18/bin/clang++",
            ],
        },
    ),
    env = {
        "CFLAGS": "-O3 -DNDEBUG  -finline-functions",
        "CXXFLAGS": "-O3 -DNDEBUG  -finline-functions",
        "AR": "/opt/homebrew/opt/llvm@18/bin/llvm-ar",
        # "ARFLAGS": "rcs",
        # "AR_FLAGS": "rcs",
        # "CC": "/opt/homebrew/opt/llvm@18/bin/clang",
        # "CXX": "/opt/homebrew/opt/llvm@18/bin/clang++",
    },
    install_prefix = "build",
    lib_source = ":openmpi_srcs",
    # targets = [
    #     "all",
    #     "install",
    # ],
    visibility = ["//visibility:public"],
)

# filegroup(
#     name = "openmpi",
#     srcs = [":configure_mpi_clang"],
#     output_group = "",
#     visibility = ["//visibility:public"],
# )
