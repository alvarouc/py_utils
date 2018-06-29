import logging
import os


class ProgressConsoleHandler(logging.StreamHandler):
    """
    A handler class which allows the cursor to stay on
    one line for selected messages
    """
    on_same_line = False

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            same_line = hasattr(record, 'same_line')
            if self.on_same_line and not same_line:
                stream.write(self.terminator)
            stream.write(msg)
            if same_line:
                stream.write('... ')
                self.on_same_line = True
            else:
                stream.write(self.terminator)
                self.on_same_line = False
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def make_logger(name='', path=None, level='INFO'):
    '''
    Make a logger object that outputs messages to
    a file and to STDOUT.

    The file format is '<path>/<name>.log'.

    Arguments:
        name - The name of the logger and the name of the file. 
               Default: 'root' (the root logger of python)
        path - The name of the path to put the logfiles in.
               Default: '~/.DISIML_logs'
        level - The minimum level at which to log. Default: 'INFO'

    Returns:
        a python logging object with the desired parameters.
    '''
    if path is None:
        path = os.path.expanduser("~/.DISIML_logs/")
    if not os.path.exists(path):
        os.makedirs(path)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    if not name:
        fh = logging.FileHandler(path + 'root.log')
    else:
        fh = logging.FileHandler(path + name + '.log')

    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    # add the handlers to the logger
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)

    elif level == 'INFO':
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # progress = ProgressConsoleHandler()
    # logger.addHandler(progress)

    return logger
