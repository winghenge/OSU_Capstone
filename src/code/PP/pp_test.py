import preproc as pp


if __name__ == "__main__":
    # create a PreProc object
    obj = pp.PreProc(save=False )

    # a simple testing loop that will take a picture everytime enter is pressed
    # when any character is entered, the program will terminate
    while(True):
        obj.capture('a')
        obj.preproccess()
        obj.cv2_disp()
        #break

    
    obj.shutdown()
