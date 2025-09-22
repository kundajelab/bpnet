import random
import string
import pytz

from datetime import datetime, timezone

def getAlphaNumericTag(length):
    """ generate a random alpha numeric tag
    
        Args:
            length (int): the desired length of the tag 
        
        Returns:
            str: alphanumeric string of length `length`
    
    """

    return ''.join(random.choices(string.ascii_letters + string.digits, 
                                  k=length))


def utc_to_local(utc, tz_str):
    """ convert utc time to a given timezone
        
        Args:
            utc (datetime.datetime): utc time
            tz_str (str): timezone string e.g. 'US/Pacific'
                
        Returns:
            datetime.datetime
    """
    
    # get time zone object from string
    tz = pytz.timezone(tz_str)
    
    return utc.replace(tzinfo=timezone.utc).astimezone(tz=tz)


def local_datetime_str(tz_str):
    """ string representation of local date & time
    
        Args:
            tz_str (str): timezone string e.g. 'US/Pacific'
            
        Returns:
            str
    """
    
    # get local datetime.datetime
    dt = utc_to_local(datetime.utcnow(), tz_str)
    
    # convert datetime.datetime to str
    return dt.strftime('%Y-%m-%d_%H_%M_%S')
