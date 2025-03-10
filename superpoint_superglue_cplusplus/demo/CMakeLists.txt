message("~~~~~~~~~~~~~~~~~~~building demo~~~~~~~~~~~~~~~~~~~~~~~~~~")


# 指定生成目标
add_executable(super_demo ${CMAKE_CURRENT_SOURCE_DIR}/super_demo.cpp
                        ${CMAKE_SOURCE_DIR}/src/superpoint.cpp
                        ${CMAKE_SOURCE_DIR}/src/superglue.cpp
                        ${CMAKE_SOURCE_DIR}/src/basic.cpp
)

target_include_directories(super_demo PUBLIC ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(super_demo ascendcl stdc++ -lopencv_imgproc -lopencv_core -lopencv_imgcodecs -lopencv_features2d dl rt)



