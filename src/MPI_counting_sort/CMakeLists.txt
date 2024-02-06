cmake_minimum_required(VERSION 3.9)

project(CommonAssignmentMPI)

add_executable(eseguibileO0 parallel.c)
add_executable(eseguibileO1 parallel.c)
add_executable(eseguibileO2 parallel.c)
add_executable(eseguibileO3 parallel.c)
add_executable(eseguibileseqO0 sequential.c)
add_executable(eseguibileseqO1 sequential.c)
add_executable(eseguibileseqO2 sequential.c)
add_executable(eseguibileseqO3 sequential.c)

target_link_libraries(eseguibileO0 PRIVATE m)
target_link_libraries(eseguibileO1 PRIVATE m)
target_link_libraries(eseguibileO2 PRIVATE m)
target_link_libraries(eseguibileO3 PRIVATE m)
target_link_libraries(eseguibileseqO0 PRIVATE m)
target_link_libraries(eseguibileseqO1 PRIVATE m)
target_link_libraries(eseguibileseqO2 PRIVATE m)
target_link_libraries(eseguibileseqO3 PRIVATE m)

find_package(MPI REQUIRED)
if(MPI_C_FOUND)
        target_link_libraries(eseguibileO0 PRIVATE MPI::MPI_C) #linko la libreria di mpi
        target_link_libraries(eseguibileO1 PRIVATE MPI::MPI_C) #linko la libreria di mpi
        target_link_libraries(eseguibileO2 PRIVATE MPI::MPI_C) #linko la libreria di mpi
        target_link_libraries(eseguibileO3 PRIVATE MPI::MPI_C) #linko la libreria di mpi
        message("MPI Trovato")
else()
    message("Non è stata trovato MPI")
endif()

target_compile_options(eseguibileO0 PRIVATE -O0) #ottimizzazioni 0
target_compile_options(eseguibileO1 PRIVATE -O1) #ottimizzazioni 1
target_compile_options(eseguibileO2 PRIVATE -O2) #ottimizzazioni 2
target_compile_options(eseguibileO3 PRIVATE -O3) #ottimizzazioni 3
target_compile_options(eseguibileseqO0 PRIVATE -O0) #ottimizzazioni 0
target_compile_options(eseguibileseqO1 PRIVATE -O1) #ottimizzazioni 1
target_compile_options(eseguibileseqO2 PRIVATE -O2) #ottimizzazioni 2
target_compile_options(eseguibileseqO3 PRIVATE -O3) #ottimizzazioni 3


# ---------------------------------- GENERATE MEASURES ----------------------------------
add_custom_target(
	 generate_measure
   COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/file.bash ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
   COMMENT "Inizio Misure"
    DEPENDS
    eseguibileseqO0
    eseguibileseqO1
    eseguibileseqO2
    eseguibileseqO3
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

add_custom_target(
    extract_measures2
    COMMAND PIPENV_PIPFILE=${CMAKE_CURRENT_SOURCE_DIR}/Pipfile ${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py
    COMMENT "Generating version header [${PipEnv} run python ${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py]"
    DEPENDS
		install_pipenv
		${CMAKE_CURRENT_SOURCE_DIR}/measures/extract.py
)
