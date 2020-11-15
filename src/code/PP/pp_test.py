import preproc as pp


if __name__ == "__main__":
    # create a PreProc object
    obj = pp.PreProc()

    # a simple testing loop that will take a picture everytime enter is pressed
    # when any character is entered, the program will terminate
    while(True):
        obj.capture('a')
        obj.preproccess()
        obj.display()
        obj.save("a")
        x = input("?")
        if x != '':
            break

    obj.shutdown()
