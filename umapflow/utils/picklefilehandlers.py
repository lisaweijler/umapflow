import pickle
from umapflow.utils.filehandler import FileLoader, FileWriter


class PickleFileLoader(FileLoader):

    def loadPickleFromInputfile(self):
        with open(self.inputfile, 'rb') as f:
            fileData = pickle.load(f)

        return fileData


class PickleFileWriter(FileWriter):

    def writePickleToOutputfile(self, data):
        with open(self.outputfile, 'wb+') as outfile:
            pickle.dump(data, outfile)

        print("INFO: successfully wrote to file: " + self.outputfile)
