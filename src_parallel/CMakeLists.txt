cmake_minimum_required (VERSION 2.8.11)

set(EXEC "qap_pga")
set(EXEC_TESTS "qap_pga_tests")

file(GLOB SRC *.cpp)

add_executable(${EXEC} ${SRC})
add_executable(${EXEC_TESTS} ${SRC})

# Ajoute la dépendence sur les fichiers clh
target_link_libraries(${EXEC} PUBLIC ${OpenCL_LIBRARY})
target_link_libraries(${EXEC_TESTS} PUBLIC ${OpenCL_LIBRARY})

target_compile_definitions(${EXEC_TESTS} PUBLIC -DTESTS)

add_custom_command(TARGET ${EXEC} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/*.cl
                       $<TARGET_FILE_DIR:${EXEC}>
                  )
