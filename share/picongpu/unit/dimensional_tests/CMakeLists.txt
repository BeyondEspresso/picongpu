# Handle tests with dimensionality
file(GLOB_RECURSE DIMENSIONAL_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
# These translation units belong to the existing TWTSTight executable, not separate tests.
set(TWTSTIGHT_SUPPORT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/twtstight/TWTSTightBesselRatiosFloat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/twtstight/TWTSTightBesselRatiosDouble.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/twtstight/TWTSTightGuard.cpp"
)
list(REMOVE_ITEM DIMENSIONAL_TESTS ${TWTSTIGHT_SUPPORT_SOURCES})
foreach(dim 2 3)
    foreach(testCaseFilepath ${DIMENSIONAL_TESTS})
        get_filename_component(testCaseFilename ${testCaseFilepath} NAME)
        string(REPLACE "UT.cpp" "" testCase ${testCaseFilename})
        set(testExe "${PROJECT_NAME}-${testCase}-${dim}D")
        set(testSources ${testCaseFilepath})
        if(testCaseFilename STREQUAL "TWTSTight.cpp")
            list(APPEND testSources ${TWTSTIGHT_SUPPORT_SOURCES})
        endif()
        alpaka_add_executable(${testExe} ${testSources})
        target_include_directories(${testExe} BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../include")
        target_include_directories(${testExe} BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_definitions(${testExe} PRIVATE TEST_DIM=${dim})
        target_link_libraries(${testExe} PUBLIC Catch2 Catch2WithMain)
        target_link_libraries(${testExe} PRIVATE pmacc::pmacc)
        add_test(
            NAME "${testCase}-${dim}D"
            COMMAND mpiexec ${MPI_RUNTIME_FLAGS} -n 1 ./${testExe}
        )
    endforeach()
    string(REPLACE "-DTEST_DIM=${dim}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endforeach()
