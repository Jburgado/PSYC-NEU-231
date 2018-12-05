
# include the import before the functions so that when we call this module later we load numpy once. 
import numpy as np

def define_filt(fc, b, window, type_filt):
    """Define a lp or a hp filter

    Args:
        fc (float): cutoff frequency, as a fraction of sampling rate
        b (float): transition band, as a fraction of sampling rate
        window (str): window sinc filter, options 'none', 'blackman', 'hanning'
        type_filt (str): low pass or high pass filter, options 'lp', 'hp'

    Returns:
        n (int array): span of filter
        sinc (float array): sinc filter in time domain

    """

    # NOTE: do some arg checks here if you want a more robust function
    
    
    # more on this below, but now need to make sure that ceil(4/b) is odd
    N = int(np.ceil((4 / b)))

    # make sure filter length is odd
    if not N % 2: N += 1  

    # generate span over which to eval sinc function    
    n = np.arange(N)

    # Compute the filter
    sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))

    # generate our window
    if window == 'blackman':
        win = np.blackman(N)
        
    elif window == 'hanning':
        win = np.hanning(N)
        
    elif window == 'none':
        # if 'none' then just an array of ones so that the values in the sinc aren't modified
        win = np.ones(N)
        
    else:
        print('Unknown window type')
    
    # apply the windowing function 
    sinc_func = sinc_func * win

    # Normalize to have an area of 1 (unit area)
    sinc_func /= np.sum(sinc_func)

    # check filter type...if lp then do nothing, else if hp invert, else return msg
    if type_filt == 'lp':
        return n, sinc_func
    
    elif type_filt == 'hp':
        # invert
        sinc_func = -1*sinc_func
        # add 1 to middle of the inverted function 
        sinc_func[int((N-1)/2)]+=1
        return n, sinc_func
    
    else:    
        print('error - specify lp or hp filter')


# Another function to apply the filter.         
def apply_filt(input_sig, input_filter):
    """Apply a filter to an input timeseries (using freq domain multiplication)

    Args:
        input_sig (float): timeseries to be filtered
        input_filter (float): filter to apply to ißnput_sig

    Returns:
        filt_sig (float array): filtered signal 

    """
    # fft our signal
    fft_sig = np.fft.rfft(input_sig)

    # need to zero pad to make the filter the same length as the signal
    X = len(input_sig)
    Y = len(input_filter)

    # zero pad in the time domain
    if Y<X:
        input_filter = np.hstack((input_filter, np.zeros(X-Y)))

    # fft the filter
    fft_filt = np.fft.rfft(input_filter)

    # multiply in freq domain, then ifft to go back into the time domain
    return np.fft.irfft(fft_sig*fft_filt)