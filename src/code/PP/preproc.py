import numpy as np
import pickle5 as pickle
from os import path
from collections import defaultdict
from scipy import ndimage
import RSC_Wrapper as RSCW
import matplotlib.pyplot as plt
import cv2
#from skimage import measure

# make the matplotlib plots interactive, this allows them to
# be updated
plt.ion()


class PreProc:
    def __init__(self, blur=False, mutate=False, scale=True, rmbkg=True,
                 resize=False, save=True):
        # This preproccessor class is intended to be used for training
        # The basic functionality of this class includes: capturing images,
        # image pre-proccessing, mutating images, and saving as a pickle

        # first, we need to load the dataset ledger, which contains the number
        # of datapoints for each gesture key
        # check to make sure the file exists (prevents against first time
        # issues)
        if (path.exists("../datasets/ds_ledger")):
            # unpickle the file and save it to this object
            with open("../datasets/ds_ledger", "rb") as fd:
                self.ledger = pickle.load(fd)
        # otherwise, create a default dict
        else:
            self.ledger = defaultdict(int)

        # now, we need to create a camera modual (RSC_Wrapper)
        self.rsc = RSCW.RSC()
        #self.rsc.start_camera()

        # alright, have an array of saved images stored before writting to disk
        self.stored_images = []

        # some pre-proccessing flags to determine the level of pre-proccessing
        # preformed to captured images
        self.f_blur = blur
        self.f_mutate = mutate
        self.f_scale = scale
        self.f_rm_background = rmbkg
        self.f_resize = resize
        self.f_save = save

    def shutdown(self):
        # safely terminate this object by saving values to disk and
        # unconnecting the camera

        # stop the camera
        self.rsc.stop_camera()

        cv2.destroyAllWindows()

        if self.f_save:
            # update the ledger
            with open("../datasets/ds_ledger", "wb") as fd:
                pickle.dump(self.ledger, fd)

    def calc_contour(self):

    def display(self):
        # update the plot/image
        plt.imshow(self.image, "gray_r")
        plt.show()

        # Matplotlib only updates the plots when the program is idle
        # thus, having the program preform a slight pause will ensure the
        # plot is updated
        plt.pause(0.001)

    def cv2_disp(self):
        cv2.imshow('Depth Image', self.image)
        cv2.waitKey(1) # waits until a key is pressed
        #time.sleep(0.0001

    def capture(self, gesture):
        # clear the saved image array, if they werent saved,
        # a user error. at this point we need to make sure
        # data is segregated
        self.stored_images.clear()

        # capture a frame from the RSC
        self.image = self.rsc.capture()

    def save(self, gesture):
        # if the save flag isnt set, DONT SAVE
        # this is mostly a testing/debugging thing so we dont have
        # data from testing this class saved
        if self.f_save:
            return
        
        # save the numpy image array as a pickle object in the training
        # database

        # now, save each image as a pickle!
        for _ in range(len(self.stored_images)):
            # create the path for the new pickle based on how many pickles
            # of that same gesture are already stored
            pth = "../datasets/" + str(gesture)+"/" + str(self.ledger[gesture])

            # Write the pickle to disk
            with open(pth, "wb") as fd:
                pickle.dump(self.stored_images.pop(), fd)

            # incriment the count in the ledger
            self.ledger[gesture] += 1

        # now clear the stored image array and the main image array
        self.stored_images.clear()
        self.image = []

    def preproccess(self):
        # Alright, based on the flags, proccess the image stored in self.image

        # first, check to see if we are going to resize, since if we are,
        # subsequent operations will be computationally cheaper
        if self.f_resize:
            self.resize()

        # remove the background (must be done before rescaling)
        if self.f_rm_background:
            self.remove_background()

        # rescale the image to be in the range of 0 to 1
        if self.f_scale:
            self.scale()

        # blur the image to reduce noise
        if self.f_blur:
            self.blur()

        # Next, we want to create mutated versions of this image for training
        if self.f_mutate:
            self.mutate()

    def remove_background(self):
        # First, find the minimum value in the depth image
        # for the depth image, the closer the object is to the camera,
        #  the smaller
        # the value. Thus, we want to find the closest pixel
        minVal = np.min(self.image[np.nonzero(self.image)])

        # Now we want to mask off the background. We do this by
        # setting any pixel thats further away than 1500 units from
        # the closest pixel to zero
        self.image[self.image > 1500 + minVal] = 0.

    def scale(self):
        # After masking off the background, find the furthest distance
        # in our ROI
        # (note, this could be minVal + 1500, but it could be smaller)
        maxVal = np.max(self.image[np.nonzero(self.image)])

        # Now we preform two operations at once, the first is to scale the ROI
        # relitive to itself
        # as such, the closest pixel should be near zero, and the furthest
        # near one.
        # Second, we raise this value to the fourth power, this is to help
        # make minor differences)
        # between pixels more distinct.
        # For example: the sign 'A' versus the sign 'S'
        # 'A' has the thumb closer to the camera than in 'S', but the
        # difference is thousanths of units. Thus, by raising to the fourth
        # power, we can increase that difference
        self.image = (self.image / maxVal) ** 4

        # Any small value (arbitrarily defined as smaller than 0.001,
        #  typically in the range ~E-5) is the
        # result of floating point errors with the masked off pixels
        # (I beleive)
        # Set these background pizels as 1, the furthest away in our range
        self.image[self.image < 0.001] = 1

    def blur(self):
        # preform a blur on the image to reduce noise
        self.image = ndimage.uniform_filter(self.image, size=10)

    def mutate(self):
        # mutate the testing image to create a larger trainging set
        # flip the image (the same same changing which hand signed,
        # L->R | R->L)
        self.image = np.flip(self.image, 1)

    def resize(self):
        # reduce the image size by 50%
        self.image = ndimage.zoom(self.image, 0.5)
