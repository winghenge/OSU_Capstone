import numpy as np
import sys
import pickle5 as pickle
from os import path, mkdir
from collections import defaultdict
import cv2
import string
from mutator import mutate


# The path to the database directory relative to this .py file
MAIN_PATH = "./database/"


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


class DB_man:
    __instance = None

    # using the __new__ magic method, along with the __instance flag
    # to ensure that only a single instance of this class is created,
    # and any subsequent calls to create a new isntance (RSC()) will return the
    # first instance. AKA, singleton
    def __new__(cls):
        if cls.__instance is None:
            # initilizer
            # first, we need to load the dataset ledger, which contains the
            # number
            # of datapoints for each gesture key
            # check to make sure the file exists (prevents against first time
            # issues)
            if (path.exists(MAIN_PATH+"ledgers/gest_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/gest_ledger", "rb") as fd:
                    cls.gest_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.gest_ledger = defaultdict(int)

            # load the dataset ledger
            if (path.exists(MAIN_PATH+"ledgers/ds_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/ds_ledger", "rb") as fd:
                    cls.ds_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.ds_ledger = 0

            # load the class ledger
            if (path.exists(MAIN_PATH+"ledgers/class_ledger")):
                # unpickle the file and save it to this object
                with open(MAIN_PATH+"ledgers/class_ledger", "rb") as fd:
                    cls.class_ledger = pickle.load(fd)
            # otherwise, create a default dict
            else:
                cls.class_ledger = {}

            cls.__instance = super(DB_man, cls).__new__(cls)

        return cls.__instance

    def save(self, image, gesture):
        pth = MAIN_PATH + str(gesture)+"/" + str(self.gest_ledger[gesture])

        # Write the pickle to disk
        with open(pth, "wb") as fd:
            pickle.dump(image, fd)

        # incriment the count in the ledger
        self.gest_ledger[gesture] += 1

        # save the updated ledger to be on the safe side
        # update the ledger
        with open(MAIN_PATH+"gest_ledger", "wb") as fd:
            pickle.dump(self.gest_ledger, fd)

    def display_from_db(self):
        # check to see if a path was provided
        if (len(sys.argv) != 2):
            print("Not enough command line arguments passed")
            return

        # load the path
        with open(sys.argv[1], "rb") as fd:
            data = pickle.load(fd)

        cv2.imshow("Data", data)
        cv2.waitKey(0)

    def load(self, gesture='all', key=True, key_type='int', split=True,
             cutoff_pcnt=.9):
        # check to see if we are going to compile all of the letters, or a
        # specific one
        if gesture == 'all':
            # compile all
            # arrays to hold the results
            x_train = []
            y_train = []
            x_test = []
            y_test = []

            # load each datapoint
            for gest in string.ascii_lowercase:
                cutoff = int(cutoff_pcnt*self.gest_ledger[gest])
                for indx in range(self.gest_ledger[gest]):
                    with open(MAIN_PATH+str(gest)+"/"+str(indx), "rb") as fd:
                        if cutoff > 0:
                            x_train.append(pickle.load(fd))
                            y_train.append(gest)
                            cutoff -= 1
                        else:
                            x_test.append(pickle.load(fd))
                            y_test.append(gest)

        elif gesture in string.ascii_lowercase: 
            #valid letter
            # arrays to hold the results
            data = []
            y_train = [gesture for _ in range(self.gest_ledger[gesture])]

            # load each datapoint
            for indx in range(self.gest_ledger[gesture]):
                with open(MAIN_PATH+gesture+"/"+str(indx), "rb") as fd:
                    data.append(pickle.load(fd))

        else:
            # not a valid gesture
            print("Gesture '" + str(gesture) + " is not in the dataset.")
            return None

        # Return Block
        if key:
            if key_type == 'ascii':
                return x_train, y_train, x_test, y_test
            else:
                for indx in range(len(y_train)):
                    y_train[indx] = ord(y_train[indx])-ord('a')
                    
                for indx in range(len(y_test)):
                    y_test[indx] = ord(y_test[indx])-ord('a')

                return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        else:
            return data

    def query_ledger(self, key, type="gesture"):
        if type == "gesture":
            return self.gest_ledger[key]
        
        elif type == "data set":
            return self.ds_ledger[key]


# now lets include some functionality allowing this file to be called by itself
# to provide some control over the database, making this into a true DBM system
if __name__ == "__main__":
    # Whoot, lets do this!
    print("Welcome to the DBM!")

    # Create a DBM object
    dbm = DB_man()

    while(1):
        # get the user's input
        x = str(input(">"))

        # if/else block to parse the user's input
        if x.lower() == "help":
            print("Commands:")
            print("Help: Display this help message")
            print("Info: Display information about the database")
            print("Discribe: Display information concerning a specific dataset")
            print("Export: Export the database as a non-mutated dataset")
            print("Mutate: Create a larger dataset from the database through image mutation")
            print("Add: Add a new gesture class to the database (not the same as Capture)")
            print("Exit: exit the DBM")
        elif x.lower() == "info":
            total = 0
            for gesture in dbm.class_ledger:
                print("Gesture ", gesture, " has ", dbm.gest_ledger[gesture], " images.")
                total += dbm.gest_ledger[gesture]
            print("Total images captured: ", total)
            
            keys = []
            for key in dbm.class_ledger:
                keys.append(key)

            print("Classes in database: ", keys)

        elif x.lower() == "add":
            # get the gesture from the user
            print("What gesture do you want to add?")
            x = str(input(">"))

            # conver the gesture to all lowercase and remove spaces/special
            # characters
            x = x.lower()
            x = x.replace(' ', '_')
            for i in '!@#$%^&*()=+':
                x = x.replace(i, '')

            # check to see if that gesture is unique
            if x not in dbm.class_ledger:
                # add x to the class ledger along with its UID,
                # and create the directory for the class
                dbm.class_ledger[x] = len(dbm.class_ledger) + 1
                mkdir(MAIN_PATH+"datasets/"+x)

                # save the updated class ledger
                with open(MAIN_PATH+"class_ledger", "wb") as fd:
                    pickle.dump(dbm.class_ledger, fd)

                print("New gesture added")

            else:
                print("That gesture is not unique. No action taken.")

        elif x.lower() == "exit":
            break
        else:
            print("Command not recognized. Use 'help' for a list of commands")
        