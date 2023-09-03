

class File():
    """
    Additional class. Speeds up interaction with files.
    """

    def __init__(self, path):

        """
        path - path to file (if we need to open it).

        :param path: str
        """

        self.path = path

    def readlines(self):

        try:
            f = open(f'{self.path}')
            raw = f.readlines()
            f.close()
            return raw

        except FileNotFoundError:
            print("Something went wrong.")

    def readline(self):

        try:
            f = open(f'{self.path}')
            raw = f.readline()
            f.close()
            return raw

        except FileNotFoundError:
            print("Something went wrong.")

    def read(self):

        try:
            f = open(f'{self.path}')
            raw = f.read()
            f.close()
            return raw

        except FileNotFoundError:
            print("Something went wrong.")

    def write(self, data):

        """
        datd - data to write to file

        :param data: str
        """

        try:
            file = open("otus.txt", "a")
            file.write(f"{data}")
            file.close()

        except FileNotFoundError:
            print("Something went wrong.")
