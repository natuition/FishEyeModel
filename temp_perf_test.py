from UndistortFishEye import fisheye_module
import cv2 as cv
import timeit

source_image_path = "images/perf_test.jpg"
calibration_file_path = "calibration_input/calibration.txt"
output_image_path = "images/undistort.png"

k, d, dims = fisheye_module.load_calibration(calibration_file_path)
img = cv.imread(source_image_path)


def undistort():
    undistorted_img = fisheye_module.undistort(img, k, d, dims)


def do_test():
    execution_time = timeit.repeat(undistort, number=1, repeat=5, globals=globals())
    print("Results:", execution_time)
    print("Min time:", min(execution_time))


if __name__ == '__main__':
    do_test()
