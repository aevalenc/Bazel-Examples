load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

filegroup(
    name = "openmpi_srcs",
    srcs = glob(["**/*"]),
)

configure_make(
    name = "configure_mpi_clang18",
    args = [
        "V=1",
        "-j4",
    ],
    # configure_in_place = True,
    configure_options = select(
        {
            "//conditions:default": [
                "CC=/usr/bin/clang-18",
                "CXX=/usr/bin/clang++-18",
            ],
            "@platforms//os:macos": [
                # "--with-libevent-libdir=/opt/homebrew/Cellar/libevent/2.1.12_1",
                # "--with-hwloc-libdir=/opt/homebrew/Cellar/hwloc/2.12.1/lib",
                "--enable-mpi-fortran=no",
                "--with-hwloc=internal",
                "--with-libevent=internal",
                "CC=/opt/homebrew/opt/llvm@18/bin/clang",
                "CXX=/opt/homebrew/opt/llvm@18/bin/clang++",
                "CFLAGS='-O3 -DNDEBUG -finline-functions'",
                "CXXFLAGS='-O3 -DNDEBUG -finline-functions'",
                "LDFLAGS=\'-Wl,-flat_namespace -Wl,-commons,use_dylibs\'",
                "RANLIB=/usr/bin/ranlib",
                "AR_FLAGS=rv",
                "AR=/usr/bin/ar",
            ],
        },
    ),
    install_prefix = "build",
    lib_source = ":openmpi_srcs",
    # out_binaries = [
    #     "mpicxx",
    #     "mpicc",
    #     "mpiexec",
    # ],
    visibility = ["//visibility:public"],
)

# filegroup(
#     name = "mpicxx",
#     srcs = [":configure_mpi_clang18"],
#     output_group = "mpicxx",
#     visibility = ["//visibility:public"],
# )

# filegroup(
#     name = "mpicc",
#     srcs = [":configure_mpi_clang18"],
#     output_group = "mpicc",
#     visibility = ["//visibility:public"],
# )

# filegroup(
#     name = "mpiexec",
#     srcs = [":configure_mpi_clang18"],
#     output_group = "mpiexec",
#     visibility = ["//visibility:public"],
# )
