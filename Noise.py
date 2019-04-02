import numpy as np
from Spectrum_Filter import *
from Filter import *
import cv2
import random


r = r'\.[a-zA-Z0-9]+'


class Noise:
    def __init__(self, method, task):
        self.method = method
        self.task = task

    def generate(self, image_name, mean=0, stddv=0):
        image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
        image = cv2.imread(image_path, 0)
        global r
        image_name = re.sub(r, '', image_name)
        height, width = image.shape
        if self.method == 'gaussian':
            for i in range(height):
                for j in range(width):
                    image[i][j] = image[i][j] + random.gauss(mean, stddv)
                    if image[i][j] < 0:
                        image[i][j] = 0
                    elif image[i][j] > 255:
                        image[i][j] = 255

        elif self.method == 'impulse_light':
            for i in range(height):
                for j in range(width):
                    temp = random.random()
                    if temp < 0.1:
                        image[i][j] = 255
                    else:
                        image[i][j] = 1.1*image[i][j]*(temp-0.1)

        elif self.method == 'impulse_dark':
            for i in range(height):
                for j in range(width):
                    temp = random.random()
                    if temp < 0.1:
                        image[i][j] = 0
                    else:
                        image[i][j] = 1.11*image[i][j]*(temp-0.1)

        else:
            image = image

        cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_{}.jpg'.
                    format(self.task, image_name, self.method), image)
        cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}_{}.bmp'.
                    format(image_name, self.method), image)


def task4():
    # Generate Gaussian Noise
    Gaussian_noise = Noise('gaussian', 4)
    Gaussian_noise.generate('lena.bmp', 30, 10)

    # Generate filter model
    Butterworth_lowpast = Filter('butterworth_lowpast', 4)
    Gaussian_lowpast = Filter('gaussian_lowpast', 4)

    # Frequency domain
    _ = Butterworth_lowpast.filtering(image_name='lena_gaussian.bmp', n=2, d0=50)
    _ = Gaussian_lowpast.filtering(image_name='lena_gaussian.bmp', n=2, d0=50)

    # Spatial domain
    spatial_filter(image_name='lena_gaussian.bmp', homework=3, task=4)


def harmonic_filter(image_name, size, Q):
    image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
    image = cv2.imread(image_path, 0)
    global r
    image_name = re.sub(r, '', image_name)
    height, width = image.shape
    for i in range(height):
        for j in range(width):
            numerator = 0
            denominator = 0
            for ii in range(size):
                for jj in range(size):
                    temp_i = int(ii - (size - 1)/2)
                    temp_j = int(jj - (size - 1)/2)
                    if temp_i+i >= height or temp_i+i < 0 or temp_j+j >= width or temp_j+j < 0:
                        numerator += 0
                        denominator += 0
                    else:
                        if image[i+temp_i][j+temp_j] == 0:
                            numerator += 0
                            denominator += 0
                        else:
                            numerator += float(image[i+temp_i][j+temp_j])**(Q+1)
                            denominator += float(image[i+temp_i][j+temp_j])**Q
            # print(numerator, denominator)
            image[i][j] = numerator/denominator

    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task5/{}_harmonic_{}.jpg'.
                format(image_name, Q), image)


def task5():
    # Generate impulse Noise
    impulse_noise_light = Noise('impulse_light', 5)
    impulse_noise_dark = Noise('impulse_dark', 5)
    impulse_noise_light.generate('lena.bmp')
    impulse_noise_dark.generate('lena.bmp')

    # Generate filter model
    Butterworth_lowpast = Filter('butterworth_lowpast', 5)
    Gaussian_lowpast = Filter('gaussian_lowpast', 5)

    # Frequency domain
    _ = Butterworth_lowpast.filtering(image_name='lena_impulse_light.bmp', n=2, d0=50)
    _ = Gaussian_lowpast.filtering(image_name='lena_impulse_light.bmp', n=2, d0=50)
    _ = Butterworth_lowpast.filtering(image_name='lena_impulse_dark.bmp', n=2, d0=50)
    _ = Gaussian_lowpast.filtering(image_name='lena_impulse_dark.bmp', n=2, d0=50)

    # Spatial domain
    spatial_filter(image_name='lena_impulse_light.bmp', homework=3, task=5)
    harmonic_filter(image_name='lena_impulse_light.bmp', size=3, Q=1)
    spatial_filter(image_name='lena_impulse_dark.bmp', homework=3, task=5)
    harmonic_filter(image_name='lena_impulse_dark.bmp', size=3, Q=1.5)
    harmonic_filter(image_name='lena_impulse_dark.bmp', size=3, Q=-1.5)
    harmonic_filter(image_name='lena_impulse_light.bmp', size=3, Q=1.5)
    harmonic_filter(image_name='lena_impulse_light.bmp', size=3, Q=-1.5)


def motion_blur(image_name, task, a, b, T):
    image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
    image = cv2.imread(image_path, 0)
    global r
    image_name = re.sub(r, '', image_name)
    height, width = image.shape
    frequency = np.fft.fft2(image)
    transformed = np.fft.fftshift(frequency)
    frequency = 20*np.log(np.abs(transformed))
    for i in range(height):
        for j in range(width):
            u = i - height/2
            v = j - width/2
            if (u * a + v * b) == 0:
                H = T
            else:
                H = T*np.sin(np.pi*(u*a+v*b))*np.exp(-1*np.pi*(u*a+v*b)*1j)
                H = H/(np.pi*(u*a+v*b))
            transformed[i][j] = transformed[i][j] * H
    filted_image = np.abs(np.fft.ifft2(transformed))
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_motion_blur.jpg'.
                format(task, image_name), filted_image)
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}_motion_blur.bmp'.
                format(image_name), filted_image)


def inverse_filtering(image_name, task, a, b, T, k):
    image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
    image = cv2.imread(image_path, 0)
    global r
    image_name = re.sub(r, '', image_name)
    height, width = image.shape
    frequency = np.fft.fft2(image)
    transformed = np.fft.fftshift(frequency)
    frequency = 20 * np.log(np.abs(transformed))
    for i in range(height):
        for j in range(width):
            u = i - height / 2
            v = j - width / 2
            if (u * a + v * b) == 0:
                H = T
            else:
                H = T * np.sin(np.pi * (u * a + v * b)) * np.exp(-1 * np.pi * (u * a + v * b) * 1j)
                H = H / (np.pi * (u * a + v * b))
                H = (np.abs(H)**2)/(H*(np.abs(H)**2 + k))
            transformed[i][j] = transformed[i][j] * H
    filted_image = np.abs(np.fft.ifft2(transformed))
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_inverse_{}.jpg'.
                format(task, image_name, k), filted_image)


def constrained_filtering(image_name, task, a, b, T, gamma):
    image_path = '/home/hero/Documents/DIP-Homework/Homework3/Requirement/{}'.format(image_name)
    image = cv2.imread(image_path, 0)
    global r
    image_name = re.sub(r, '', image_name)
    height, width = image.shape

    # Generate P(u, v)
    laplace_model = np.zeros((height, width))
    i = int((height-1)/2)
    j = int((width-1)/2)
    laplace_model[i, j] = 4
    laplace_model[i - 1, j - 1] = -1
    laplace_model[i + 1, j - 1] = -1
    laplace_model[i - 1, j + 1] = -1
    laplace_model[i + 1, j + 1] = -1
    laplace_frequency = np.fft.fft2(laplace_model)
    laplace_transformed = np.fft.fftshift(laplace_frequency)

    # Do fft
    frequency = np.fft.fft2(image)
    transformed = np.fft.fftshift(frequency)

    # Filtering
    for i in range(height):
        for j in range(width):
            u = i - height / 2
            v = j - width / 2
            if (u * a + v * b) == 0:
                H = T
            else:
                H = T * np.sin(np.pi * (u * a + v * b)) * np.exp(-1 * np.pi * (u * a + v * b) * 1j)
                H = H / (np.pi * (u * a + v * b))
                H = H.conjugate()/(np.abs(H)**2 + gamma * (np.abs(laplace_transformed[i][j])**2))
            transformed[i][j] = transformed[i][j] * H
    filted_image = np.abs(np.fft.ifft2(transformed))
    cv2.imwrite('/home/hero/Documents/DIP-Homework/Homework3/Content/task{}/{}_constrained_{}.jpg'.
                format(task, image_name, gamma), filted_image)


def task6():
    a, b, T = 0.1, 0.1, 1
    # Motion Blur
    motion_blur('lena.bmp', 6, 0.1, 0.1, 1)
    # Add gaussian noise
    Gaussian_noise = Noise('gaussian', 6)
    Gaussian_noise.generate('lena_motion_blur.bmp', 0, np.sqrt(10))
    # Restore the image
    inverse_filtering('lena_motion_blur_gaussian.bmp', 6, a, b, T, 0.015)
    constrained_filtering('lena_motion_blur_gaussian.bmp', 6, a, b, T, 0.015)


if __name__ == '__main__':
    # task4()
    # task5()
    task6()

