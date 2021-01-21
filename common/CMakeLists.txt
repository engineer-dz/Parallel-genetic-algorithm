file(GLOB include_Common *.h *.hpp *.c)

install(FILES ${include_Common} DESTINATION common)
