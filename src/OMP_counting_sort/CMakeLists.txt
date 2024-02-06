cmake_minimum_required(VERSION 3.9)

project(CMakeTestProject2)

add_executable(eseguibileO0 main.c)
add_executable(eseguibileO1 main.c)
add_executable(eseguibileO2 main.c)
add_executable(eseguibileO3 main.c)


find_package(OpenMP)
if(OpenMP_C_FOUND)
        target_link_libraries(eseguibileO0 PRIVATE OpenMP::OpenMP_C) #linko la libreria -fopenmp
        target_link_libraries(eseguibileO1 PRIVATE OpenMP::OpenMP_C) #linko la libreria -fopenmp
        target_link_libraries(eseguibileO2 PRIVATE OpenMP::OpenMP_C) #linko la libreria -fopenmp
        target_link_libraries(eseguibileO3 PRIVATE OpenMP::OpenMP_C) #linko la libreria -fopenmp
        message("OpenMP Trovato")
else()
    message("Non è stata trovato OpenMP")
endif()

target_compile_options(eseguibileO0 PRIVATE -O0) #ottimizzazioni 0
target_compile_options(eseguibileO1 PRIVATE -O1) #ottimizzazioni 1
target_compile_options(eseguibileO2 PRIVATE -O2) #ottimizzazioni 2
target_compile_options(eseguibileO3 PRIVATE -O3) #ottimizzazioni 3


add_custom_target(
    generate_output
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/file.bash ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Inizio Misure"
    DEPENDS	
	eseguibileO0
        eseguibileO1
        eseguibileO2
        eseguibileO3
        
)

# --------------------------------- MEASURES EXTRACTION ---------------------------------
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(PipEnv pipenv REQUIRED)

add_custom_target(
	 install_pipenv
	 COMMAND PIPENV_PIPFILE=${CMAKE_CURRENT_SOURCE_DIR}/Pipfile ${PipEnv} install --ignore-pipfile
    COMMENT "Installing pipenv dependencies [${PipEnv} install]"
    DEPENDS	${CMAKE_CURRENT_SOURCE_DIR}/Pipfile.lock
)

add_custom_target(
    extract_measures
    COMMAND PIPENV_PIPFILE=${CMAKE_CURRENT_SOURCE_DIR}/Pipfile ${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/estrai.py
    COMMENT "Generating version header [${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py]"
    DEPENDS	
		install_pipenv
		${CMAKE_CURRENT_SOURCE_DIR}/estrai.py
)
