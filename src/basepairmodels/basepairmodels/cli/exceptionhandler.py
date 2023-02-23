"""
A custom class for exceptions without printing the traceback
"""

from __future__ import print_function
import sys


def eprint(*args, **kwargs):
    """ 
        print function to print to standard error
    """
    print(*args, file=sys.stderr, **kwargs)


class NoTracebackException(Exception):
    """
        An exception that when raised results in the error message
        being printed without the traceback
    """
    pass


def notraceback_hook(kind, message, traceback):
    """
        Exception hook that reroutes all exceptions through this method
    
        Args:
            kind (type): the type of the exception
            message (obj): the exception instance
            traceback (traceback): traceback object
    """
   
    if kind.__name__ == "NoTracebackException":
        # only print message
        eprint('ERROR: {}'.format(message))  
    else:
        # print error type, message & traceback
        sys.__excepthook__(kind, message, traceback)  


# customize handling of exceptions by assigning the exception hook
sys.excepthook = notraceback_hook
