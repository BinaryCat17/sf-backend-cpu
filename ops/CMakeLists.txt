# --- Kernel Generation ---
find_package(Python3 REQUIRED)

set(SF_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${SF_GEN_DIR}")

set(CPU_SPEC "${CMAKE_CURRENT_SOURCE_DIR}/../tools/metadata/cpu_spec.json")
set(KERNELS_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/../tools/templates/sf_ops_cpu_auto.c.j2")
set(KERNELS_C "${SF_GEN_DIR}/sf_ops_auto.c")
set(DISPATCH_TMPL "${CMAKE_CURRENT_SOURCE_DIR}/../tools/templates/sf_ops_auto_dispatch.c.j2")
set(DISPATCH_C "${SF_GEN_DIR}/sf_ops_core.c")

add_custom_command(
    OUTPUT "${KERNELS_C}" "${DISPATCH_C}"
    COMMAND Python3::Interpreter "${SF_ISA_GEN_SCRIPT}" --isa "${SF_ISA_METADATA}" --backend "${CPU_SPEC}"
            --render "${KERNELS_TMPL}:${KERNELS_C}"
                     "${DISPATCH_TMPL}:${DISPATCH_C}"
    DEPENDS "${SF_ISA_METADATA}" "${CPU_SPEC}" "${SF_ISA_GEN_SCRIPT}" "${KERNELS_TMPL}" "${DISPATCH_TMPL}"
    COMMENT "Generating CPU Kernels and Dispatch Table"
    VERBATIM
)

add_library(ops STATIC
    src/sf_ops_array.c
    "${DISPATCH_C}"
    src/sf_ops_math.c
    src/sf_ops_logic.c
    src/sf_ops_matrix.c
    src/sf_ops_state.c
    src/sf_ops_system.c
    "${KERNELS_C}"
)
add_library(SionFlow::ops ALIAS ops)

target_include_directories(ops 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src
)

target_link_libraries(ops 
    PUBLIC 
        SionFlow::isa
        m
)
