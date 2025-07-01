load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@rules_cc//cc:defs.bzl", "cc_binary")

bool_flag(
    name = "use_secondary_json_file",
    build_setting_default = False,
)

config_setting(
    name = "use_secondary_json_file_config",
    flag_values = {"//using_select:use_secondary_json_file": "True"},
)

cc_binary(
    name = "using_select",
    srcs = ["using_select.cpp"],
    data = select({
        ":use_secondary_json_file_config": [":secondary_input.json"],
        "//conditions:default": [":primary_input.json"],
    }),
    defines = select({
        ":use_secondary_json_file_config": ["USE_SECONDARY_JSON_FILE"],
        "//conditions:default": ["USE_DEFAULT_CONFIG"],
    }),
    deps = ["@nlohmann_json//:json"],
)
