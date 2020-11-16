import preproc as pp
import RSC_Wrapper as RSCW


if __name__ == "__main__":
    # create a PreProc object
    obj = pp.PreProc(save=False)
    obj2 = pp.PreProc()

    rsc1 = RSCW.RSC()
    rsc2 = RSCW.RSC()

    print(obj is obj2)
    print(rsc1 is rsc2)

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
