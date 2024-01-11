# for debug printing
from icecream import ic



def safe_inv(x:float) -> float:
    """Avoid divide by zero land for 1/x """
    C = 299792458.0
    if x == 0.0:
        output = 1.0
    else:
        output = 1.0/x
    if output >= C:
        return C
    else:
        return output
    
    
def cap_c(x: float) -> float:
    """Cap values at c because fuck you."""
    C = 299792458.0
    if x >= C:
        return C
    else:
        return x


def safe_div(x:float, y: float) -> float:
    """Avoid divide by zero land for x/y"""
    C = 299792458.0
    if y == 0.0:
        y = 1.0 # jump up to 1.0 instead of divide by zero error.
    else:
        y = y
    output = x/y
    if output >= C:
        return C
    else:
        return output
    
    

def psi(e: float) -> float:
    """
    One solution of the Schroedinger equation:
    psi(e) = A * e^((+/-i * sqrt(2mE) * x)/h)
    'plus or minus' means: if it's in [0,1] and hits 1, go neg, if it hits 0, go pos
    Reminder: imaginary numbers in Python use the suffix -j, like 1j, 2.5j, 3j...
    component waves can have these forms:
    1) 
    2) 
    """
    import math.e as e
    import math.sqrt
    A = 1 # find constant later if needed
    h = 1 # set to Planck constant
    
    
    

def calculate_psi(time, xo, k, s):
    """
    Adapted from https://www.taheramlaki.com/blog/articles/quantum-particles/
    xo <- initial wave packet's center, x0
    k <- initial momentum, k0
    s <- sigma
    Returns NumPy ndarray of complex numbers.
    """
    import numpy as np
    x_min, x_max, num_points = -20, 20, 8000 # leads to an array of 8000 points evenly spaced in [-20, 20]
    grid = np.linspace(x_min, x_max, num_points) # this creates that array, it's a grid for e.g. matplotlib 
    x_c, ko, sigma = -15, 2, 1.0 # starting x coordinate in the plot interval, starting momentum, starting sigma
    s2 = s**2 # square of the sigma value
    # Computing the equation's return value has 3 steps
    psi = np.exp(-0.25 * np.square(grid - xo - 2j * k * s2) / (s2 + 1j * time)) # step 1
    psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * time) # step 2
    psi = np.power(0.5 * s2 / np.pi, 0.25) * psi # step 3
    return psi

test = calculate_psi(1, 1, 1, 1)
ic(test)





def static_packet(x: float, s: float):
    """
    Partial of calculate_psi() for static wave packet.
    x <- position
    s <- sigma
    Returns NumPy ndarray of complex numbers.
    """
    t = 0 # zero time value for static packet
    k = 0 # static packet has zero momentum
    psi = calculate_psi(t, x, k, s)
    return psi



def moving_packet_new(t: list[float], x: list[float], k: float, s: float):
    """
    The speed of light thing Einstein found from Maxwell's equations means dx/dt cannot exceed C.
    A rough solution is to normalize everything to [0, C] after finding the rates and then generate new position and time lists.
    
    Time is input as a list whose values are arbitrary.
    
    X positions are input as a list, also. Both X and T are subject to change.
    
    Returns NumPy matrix.
    
    ~~~
    About one of the functions.
    numpy.gradient(array-like) returns a numpy array of the same contour indicating differences.
    
    Basically it is y/dx or a derivative.
    
    In: np.gradient([1,2,3])
    Out: array([1., 1., 1.])
    
    In: np.gradient([1,5,7,1])
    Out: array([ 4.,  3., -2., -6.])
    ~~~
    
    This all outputs a NumPy matrix. Each row in the matrix is the output of calculate_psi() for constrained dx/dt values.
    
    This could be used to update an animation of a changing wave packet, but should respect relativistic limits.
    """
    import numpy as np
    from sklearn import preprocessing
    #from scipy import interpolate
    #interp = interpolate.interp1d # interp(x,y) interpolate a numerical function y over a new x range in 1D
    # x and y arrays must be equal in length along original interpolation axis, then new function consumes new x axis.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    min_max_scaler = preprocessing.MinMaxScaler() # normalizes matrix values to unit vector
    # usage is min_max_scaler.fit_transform(numpy_matrix)
    
    derivative = np.gradient # f: list[float] -> list[float]
    integral = np.trapz # f: list[float] -> float
    
    def mv_pkt(T, X):
        """Partial function over psi for constant momentum k and constant sigma s."""
        K = k
        S = s
        return calculate_psi(T, X, K, S)
    
    # time values gradient
    t_grad = derivative(t)
    # x values gradient
    x_grad = derivative(x)
    # d is the x/t gradient
    d = []
    if len(t) == len(x):
        for i in range(len(t)):
            d.append[safe_div(x,t)]
    else:
        length = min(len(t), len(x))
        for i in range(length):
            d.append[safe_div(x,t)]
    # Compute gradient of x/t values which is hopefully an equal length list roughly equivalent to (x1-x0)/(t1-t0) for each value.
    d_grad = np.gradient(d) # this should be all of the dx/dt values in the range.
    
    # Gradient of gradients
    g_grad = np.gradient([safe_div(a,b) for (a,b) in zip(x_grad, t_grad)])
    ic(g_grad) # debug
    
    # Some ways to advance new values for x over time delta.
    new_x_one = [xi+di for (xi,di) in zip(x, d_grad)]
    new_x_two = [xi+dx+di for (xi,dx,di) in zip(x, x_grad, d_grad)]
    new_x = new_x_one
    
    # Compute a matrix of psi contours for each value in t and new x values
    psi_matrix = np.array(mv_pkt(ti, xi) for (ti,xi) in zip(t, new_x))
    return psi_matrix


def moving_packet_old(t: list[float], x: list[float], k: float, s: float) -> tuple:
    """
    The source material assumes the moving packet is a photon, i.e. no time value.
    This definition needs a limiter that prevents dx/dt from exceeding c
    If the value would exceed c, then t increases until the value is within range.
    Returns NumPy ndarray of complex numbers.
    """
    # Could replace ths limiter thing to just normalize the deltas to [0, C]
    def limiter(t: list[float], x:list[float]) -> list[float]:
        """
        Write something that prevents dx/dt from exceeding c = 299792458, conceived of somehow.
        Consumes an array of time values and an array of x axis positions.
        itertools.pairwise is available from Python 3.10 onward.
        """
        from itertools import pairwise
        import numpy as np
        C = 299792458.0
        # These two values should be the same.
        len_t = len(t)
        len_x = len(x)
        ic(len_t, len_x, len_t - len_x) # debug
        diffs_x = [a-b for (a,b) in pairwise(x)]
        ic(diffs_x) # debug
        diffs_t = [a-b for (a,b) in pairwise(t)]
        ic(diffs_t) # debug
        deltas = [dx/dt for (dx,dt) in zip(diffs_x, diffs_t)]
        ic(deltas) # debug
        # Look for anything that exceeds c
        i = 0
        while i < len(deltas):
            if deltas[i] >= C:
                deltas[i] = C
            else:
                deltas[i] = deltas[i]
            i += 1
        d_min, d_max = np.min(deltas), np.max(deltas)
        grid_t = np.linspace(d_min, d_max, len_t) 
        grid_x = np.linspace(d_min, d_max, len_x)
        # Generate list of same length as x and t inputs but spanning min and max
        
        return t, x
    
    tx, kx = limiter(t, x)
    psi = calculate_psi(t, x, k, s)
    return psi


def photon(x: float, k: float, s: float) -> float:
    t = 0 # no time delta for photons
    psi = calculate_psi(t, x, k, s)
    return psi