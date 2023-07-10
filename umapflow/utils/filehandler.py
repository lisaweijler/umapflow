import os
from abc import ABC


class FileLoader(ABC):

    def __init__(self, inputfile: str):

        if inputfile == "":
            raise ValueError("inputfile is empty")

        self.checkFileExists(inputfile)

        self.inputfile = inputfile

    def checkFileExists(self, inputfile):
        if not os.path.exists(inputfile):
            print(inputfile + " does not exist")
            raise Exception(inputfile + " does not exist")


class FileWriter(ABC):

    def __init__(self, outputfile):

        if outputfile == "":
            raise ValueError("outputfile is empty")

        # self.checkFileDoesNotExists(outputfile)

        self.outputfile = outputfile

    def checkFileDoesNotExists(self, outputfile):
        if os.path.exists(outputfile):
            print(outputfile + " does exist")
            raise Exception(outputfile + " already exist")
