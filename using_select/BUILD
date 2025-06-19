load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("@rules_cc//cc:defs.bzl", "cc_binary")

string_flag(
    name = "json_config",
    build_setting_default = "lol",
)

USE_DEFAULT_CONFIG = ":primary_input.json"

config_setting(
    name = "use_secondary_json_file",
    flag_values = {"//using_select:json_config": "use_secondary_json_file"},
)

cc_binary(
    name = "using_select",
    srcs = ["using_select.cpp"],
    data = select({
        "//conditions:default": [USE_DEFAULT_CONFIG],
        ":use_secondary_json_file": [":secondary_input.json"],
    }),
)
