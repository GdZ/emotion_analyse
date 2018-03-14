import time


class logger:

    def __init__(self, config):
        self.config = config
        self.info = self.config.INFO
        self.debug = self.config.DEBUG
        self.warn = self.config.WARN
        self.error = self.config.ERROR
        self.fatal = self.config.FATIL

    def d(self, msg):
        if self.debug:
            print("[release] %s %s" % (time.time(), msg))
        else:
            pass

    def i(self, msg):
        if self.info:
            print("[info] %s %s" % (time.time(), msg))
        else:
            pass

    def w(self, msg):
        if self.warn:
            print("[warn] %s %s" % (time.time(), msg))
        else:
            pass

    def e(self, msg):
        if self.error:
            print("[warn] %s %s" % (time.time(), msg))
        else:
            pass

    def f(self, msg):
        if self.fatal:
            print("[fatal] %s %s" % (time.time(), msg))
        else:
            pass

    def is_debug(self):
        return self.debug
