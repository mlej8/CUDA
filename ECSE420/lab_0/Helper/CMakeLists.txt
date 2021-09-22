# libraries
add_library(image_io STATIC lodepng.cpp)
add_library(image_equality STATIC image_equality.cpp)
target_link_libraries(image_equality PRIVATE image_io)

add_executable(test_image_quality test_image_quality.cpp)
target_link_libraries(test_image_quality PRIVATE image_equality)
add_executable(loadpng_test lodepng_test.cpp)
target_link_libraries(loadpng_test PRIVATE image_io)