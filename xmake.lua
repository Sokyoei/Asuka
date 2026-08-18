set_xmakever("2.3.0")

set_project("Asuka")
set_version("0.0.1")

add_rules("mode.debug", "mode.release")
set_languages("c17", "c++20")

------------------------------------------------------------------------------------------------------------------------
-- options
------------------------------------------------------------------------------------------------------------------------
option("learning",          { default = false, showmenu=true, description = "build learning subproject"         })

------------------------------------------------------------------------------------------------------------------------
-- global settings
------------------------------------------------------------------------------------------------------------------------
add_defines("AHRI_EXPORT")

set_warnings("all")

-- include dir
add_includedirs("$(projectdir)")
add_includedirs("$(projectdir)/include")

-- config.h
set_configdir("$(projectdir)")
add_configfiles("config.h.xmake", { filename = "config.h" })
set_configvar("ASUKA_ROOT", (function()
    projectdir, count = string.gsub(os.projectdir(), "\\", "/")
    return projectdir
end)())

------------------------------------------------------------------------------------------------------------------------
-- compiler and linker
------------------------------------------------------------------------------------------------------------------------
-- GCC
add_cxflags("-fdiagnostics-color=always", { tools = "gcc" })
if is_os("windows") then
    add_cxflags("-fexec-charset=gbk", { tools = "gcc" })
end
-- MSVC
add_cxflags("/EHsc", "/utf-8", { tools = "cl" })
add_cflags("/Zc:__STDC__", { tools = "cl" })
add_cxxflags("/Zc:__cplusplus", { tools = "cl" })
add_cxxflags("/experimental:module", { tools = "cl" })
add_ldflags("/subsystem:console")

------------------------------------------------------------------------------------------------------------------------
-- libraries
------------------------------------------------------------------------------------------------------------------------
add_requires("gtest", { configs = { main = true, shared = true, gmock = true } })
if has_package("gtest") then
    set_configvar("ASUKA_USE_GTEST", true)
end
add_requires("fmt", { configs = { header_only = true } })
if has_package("fmt") then
    set_configvar("ASUKA_USE_FMT", true)
end
add_requires("spdlog", { configs = { header_only = true } })
if has_package("spdlog") then
    set_configvar("ASUKA_USE_SPDLOG", true)
end
add_requires("nlohmann_json")
if has_package("nlohmann_json") then
    set_configvar("ASUKA_USE_NLOHMANN_JSON", true)
end
add_requires("tinyxml2")
if has_package("tinyxml2") then
    set_configvar("ASUKA_USE_TINYXML2", true)
end
-- add_requires("toml11")
-- if has_package("toml11") then
--     set_configvar("ASUKA_USE_TOML11", true)
-- end
add_requires("toml++")
if has_package("toml++") then
    set_configvar("ASUKA_USE_TOMLPLUSPLUS", true)
end
add_requires("yaml-cpp")
if has_package("yaml-cpp") then
    set_configvar("ASUKA_USE_YAML_CPP", true)
end
add_requires("boost")
if has_package("boost") then
    set_configvar("ASUKA_USE_BOOST", true)
end
add_requires("opencv")
if has_package("opencv") then
    set_configvar("ASUKA_USE_OPENCV", true)
end

------------------------------------------------------------------------------------------------------------------------
-- subdir
------------------------------------------------------------------------------------------------------------------------
if has_config("learning") then
    includes("learning")
end
