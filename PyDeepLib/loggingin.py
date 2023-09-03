import logging

class Logging():
    """
    Additional class. Responsible for logging.
    Logging only within library files.
    """

    def __init__(self, logging=False, level='DEBUG', file='PDLibs.log'):

        """
        logging - logging state (enabled or disabled).
        level - level of logging (the main logging occurs at the error levels and below,
                the remaining levels in this case are meaningless and are not used).
        file - file name for logging.

        :param logging: bool
        :param level: str
        :param file: str
        """

        self.logging = logging
        self.level = level
        self.file = file

    def log(self):

        try:

            if self.logging:

                if self.level == 'DEBUG' or self.level == 'debug':
                    logging.basicConfig(level=logging.DEBUG, filename=f"{self.file}", filemode="w")

                elif self.level == 'INFO' or self.level == 'info':
                    logging.basicConfig(level=logging.INFO, filename=f"{self.file}", filemode="w")

                elif self.level == 'ERROR' or self.level == 'error':
                    logging.basicConfig(level=logging.ERROR, filename=f"{self.file}", filemode="w")

                logging.error("[+] Start logging. Logging was start successfully.")

            else:
                pass

        except Warning:
            print("Error while working with logs.")
