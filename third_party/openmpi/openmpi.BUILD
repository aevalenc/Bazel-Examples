genrule(
    name = "configure_mpi_clang18",
    srcs = glob(["**/*"]),
    outs = ["setup.out"],
    cmd = """
    echo "mkdir -p build" > $@
    echo "touch build/setup.out" >> $@
    """,
    executable = True,
    visibility = ["//visibility:public"],
)
