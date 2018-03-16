import time
import traceback


class logger:

    def __init__(self, config):
        self.config = config
        self.info = self.config.INFO
        self.debug = self.config.DEBUG
        self.warn = self.config.WARN
        self.error = self.config.ERROR
        self.fatal = self.config.FATAL

    def d(self, msg):
        if self.debug:
            print("[debug] %s %s" % (time.time(), msg))
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

    def e(self, Exception, e):
        if self.error:
            print("[error] %s %s" % (time.time(), e.message))
            print 'str(Exception):\t', str(IndexError)
            print 'str(e):\t\t', str(e)
            print 'repr(e):\t', repr(e)
            print 'e.message:\t', e.message
            print 'traceback.print_exc():'; traceback.print_exc()
            print 'traceback.format_exc():\n%s' % traceback.format_exc()
        else:
            pass

    def f(self, msg):
        if self.fatal:
            print("[fatal] %s %s" % (time.time(), msg))
        else:
            pass

    def is_debug(self):
        return self.debug
