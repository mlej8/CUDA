# libraries
add_library(image_io STATIC lodepng.cpp lodepng.hpp)
add_library(image_equality STATIC image_equality.cpp lodepng.hpp)
target_link_libraries(image_equality PRIVATE image_io)

add_executable(test_image_equality test_image_equality.cpp image_equality.hpp)
target_link_libraries(test_image_equality PRIVATE image_equality)
add_executable(loadpng_test lodepng_test.cpp lodepng.hpp)
target_link_libraries(loadpng_test PRIVATE image_io)