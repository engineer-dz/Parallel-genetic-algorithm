cmake_minimum_required (VERSION 2.8.11)

set(EXEC "qap_pga")

file(GLOB SRC *.cpp)

add_executable(${EXEC} ${SRC})

# Ajoute la dépendence sur les fichiers clh
target_link_libraries(${EXEC} PUBLIC ${OpenCL_LIBRARY})

add_custom_command(TARGET ${EXEC} POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/generate_individual.cl
                       $<TARGET_FILE_DIR:${EXEC}>
                  )
