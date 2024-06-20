cmake_minimum_required(VERSION 3.13)

project(
    tskit
    DESCRIPTION "Succinct tree sequences"
    LANGUAGES C
)

# -- warning flags --
list(
    APPEND
    TSKIT_C_FLAGS
    "-Wall"
    "-Wextra"
    "-Werror"
    "-Wpedantic"
    "-W"
    "-Wmissing-prototypes"
    "-Wstrict-prototypes"
    "-Wconversion"
    "-Wshadow"
    "-Wpointer-arith"
    "-Wcast-align"
    "-Wcast-qual"
    "-Wwrite-strings"
    "-Wnested-externs"
    "-fshort-enums"
    "-fno-common"
)

# -- kastore --
add_library(kastore STATIC)
target_compile_features(kastore PRIVATE c_std_99)
set_target_properties(kastore PROPERTIES LINKER_LANGUAGE C)
target_compile_options(kastore PRIVATE ${TSKIT_C_FLAGS})
target_include_directories(kastore PUBLIC "subprojects/kastore")
target_sources(
    kastore
    PRIVATE subprojects/kastore/kastore.c
    PUBLIC subprojects/kastore/kastore.h
)

# -- tskit --
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_library(tskit STATIC)
target_compile_features(tskit PRIVATE c_std_99)
set_target_properties(tskit PROPERTIES LINKER_LANGUAGE C)
target_compile_options(tskit PRIVATE ${TSKIT_C_FLAGS})
target_include_directories(tskit PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_sources(
    kastore
    PRIVATE tskit/convert.c
            tskit/core.c
            tskit/genotypes.c
            tskit/haplotype_matching.c
            tskit/stats.c
            tskit/tables.c
            tskit/trees.c
    PUBLIC tskit/convert.h
           tskit/core.h
           tskit/genotypes.h
           tskit/haplotype_matching.h
           tskit/stats.h
           tskit/tables.h
           tskit/trees.h
)
target_link_libraries(tskit kastore m)

