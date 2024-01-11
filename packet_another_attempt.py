__copyright__ = """
Copyright: TAKEMITSU, Zeami (武満世阿弥) "Brien"

"Reach out to Kyoto University with my Japanese name!"

My family is from Japan. I was in Marseille processing trauma from the Foreign Legion where I served for 5 years.
I was losing sleep at night thinking about physics. War is physics. I used to do Physics before it was War.

How can I know if the peaks and troughs inside a quantum wave packet are traveling faster than light or not?
Other waves I have observed in nature are subject to physical constraints when they propagate information. It should be that way all over.
The waves shouldn't go FTL. No crest or trough can go faster than the standing wave body without it ceasing to be a wave.

But a wave packet is hard to examine. Still, why shouldn't I assume a wave function has an inertial frame of reference
that's subject to Lorenz boosts, blue shift / red shift, etc like any other stationary or traveling wave under relativity?

Limits on the propagation of information through light are limits on the behavior of information. I am speculating that this could lead to
a novel machine learning approach because the information contour of the universe is being modeled by learning machines. So that contour
might be as below, so above, in that conversation laws, FTL limits, etc can apply to all Joules, even information theoretic ones.

So if information is subject to Lorenz boosts, conversation laws, phenomena from statistical mechanics, etc then all of this must also
be subject to inertial frames of reference, because other waves exhibit the same phenomena, and this limits our information about the
state of physical systems.

Inertial frame of reference conservation laws applying to the behavior of traveling Quantum waves could potentially resolve certain
unanswered questions in Physics and even Mathematics.

Notion for ML application: The quantum wave packet function discretizes information and expands its representation into the complex domain. 
Maybe it can help map the contours of an entropic system to use the complex domain.

Machine learning systems are contour-modeling entropy wells for estimation. Rates of change model contours. Those are subject to FTL limits.

This combination of factors has potential to trap entropy in a way that's substantially different from neural nets.

For machine learning: 

Recursion and function optimizer are to be used until it predicts outputs from inputs well enough.

* Quantum wave function in complex domain, instead of visualizing it fills a variable-size matrix.
* Recursive Monadic Interpolation -- apply linear interpolator over continuously changing matrix size and track state in a monad. Optimize.
** Need a way to track the summarization and extrapolation of information like in different-sized layers within neural nets. Monad-consuming fuctors?
** We compute the contents of matrix dimensionalities using wave packets, currying / monads, and recursive information extraction.
** This will interpolate across different matrix dimensionalities, using that to make inferences from inputs. The matrices are squares.
** For example, the input is 100x100 but we want guesses at a smaller projection of the wave packet but the matrix is e x e dimensions or pi x pi dimensions.
* Predicted values are from contours in SVTs and other systems, this does contours and matrices.
** The contours and matrices are built around the behavior of quantum wave packets.
* Gonna leave the Schroeinger equation solution in this as a way to map information about inputs. 
** It creates a wave packet. That's the point of the visualizations.
** Wave packets are characteristic of systems. They model information already.

For physics: 

Need to determine if wave body compression within the packet is testable by e.g. perpendicular entangled lasers.
Need to include a basic mathematical proof for some people that a wave's peaks and troughs cannot go faster than its body can.
That seems simple but it absolutely implies an inertial frame of reference applying to the sine operator in a QM wave packet, not just
the sine operator applying to waves at sea, or kinetic waves in the air, or waves moving through solid matter. The crest and the peak
do not beat the wave body in reaching a point, they constitute its nature as a wave.

That this necessarily implies inertial frames of reference and Lorenz boosts applying to wave functions is called the 
Kyoto-Marseille Interpretation, because I must have to be stupid to propose a name for a physics interpretation.

Some physicists ignore that the Joule is used to measure both information and energy in a way that implies a system-dependent coefficient
that itself is frequently lower in highly entropic systems, but only to a point. J1 = K * J2, where J1 is information output out of a
computational system in Joules and J2 is energy input into the computer in Joules. Computational complexity is thermodynamic in nature.

So information propagating about this idea is going to be limited by the same physical constraints and can only occur in a seemingly
FTL way through wormholes / 'portals'. Necessarily an extension of this viewpoint about waves, should it be accepted, is the presumption
of compactified wormholes permitting information transfer in violation of known 'boosts' and conservation laws.
"""

import numpy as np
gradient = np.gradient # f: array -> array
derivative = np.gradient # f: array -> array
second_deriv = lambda x: gradient(gradient(x))[0] # f: array -> float
integral = np.trapz # f: array -> float
# This linear interpolator accepts periodic values
interpolate = np.interp # f: value or array, array, array -> value or array

from sklearn import preprocessing
# normalizes matrix values to unit vector
min_max_scaler = preprocessing.MinMaxScaler()

# Memoizer. Does not work with hashables but works with iterables.
from functools import lru_cache

# not finding the library for prettier debug prints???
#from icecream import ic
ic = print


# the new domain input can be periodic, all that matters is that x_list and y_list are the same length
new_domain, x_list, y_list = [0, 0.1, 10, 0.2, 0], [0,1,2,3], [0,2,4,6]
ic(interpolate(new_domain, x_list, y_list))


# Let's test with a sinusoid over 1000 steps.
time = np.linspace(0.0, 1.0, 1000)
sinusoid = np.sin(time * 2 * np.pi)

new_time = np.linspace(0.0, 1.0, 10000) # 10x as many steps in the x axis to interpolate y values over
sin_interp = interpolate(new_time, time, sinusoid)


@lru_cache
def cap_c(x: float) -> float:
    C = 299792458.0
    if x >= C:
        return C
    else:
        return x
    

@lru_cache
def cap_pl(x: float) -> float:
    # C as measured in m/s
    C = 299792458.0
    h = 6.62607015e-34 # Planck's constant value
    G = 6.6743e-11 # gravitational constant value
    planck_length = np.sqrt((h*G) / C**3)
    if x < planck_length:
        return planck_length
    else:
        return x
    


def limit_delta(x_list, t_list):
    """
    The delta between x0 and x1 over the interval t0 to t1 should not exceed C
    Steps should also not exceed the Planck length in the opposite direction.
    The two inputs must be the same length.
    Returns dx/dt where the max value is capped at the speed of light in m/s
    """
    x_grad = gradient(x_list)
    t_grad = gradient(t_list)
    d_grad = x_grad / t_grad
    capped = np.array([cap_c(d) for d in d_grad]) # keep Dx steps below light speed
    capped = np.array([cap_pl(d) for d in capped]) # keep Dx steps above minimum Planck length
    return capped
    

def lorentz_boost(x: list[float], t: list[float], v: list[float]) -> tuple:
    """
    Returns new tuple of (dx, dt, dv) from Lorentz Transform. 
    
    The function defined below this one limit_delta() returns a list of velocity values from X and T inputs
    from within the relativistic limiting that occurs. That can feed this one to advance time values.

    Args:
        t (list[float]): list of Time values
        x (list[float]): list of position values on an axis (defaults to X in naming)
        v (list[float]): list of velocity values; don't leave out because this can reflect higher order rates of change

    Returns:
        tuple[list[float]]: transformed values.
    """
    import numpy as np
    
    # C as measured in m/s
    C = 299792458.0
    C2 = C**2
    
    h = 6.62607015e-34 # Planck's constant value
    G = 6.6743e-11 # gravitational constant value
    planck_length = np.sqrt((h*G) / C**3)
    
    ly = 9.461e+15 # lightyear in meters
    
    inv_planck_length = 1. / planck_length
    
    # Largest integer Python 3.12 can support as a string, as an integer
    big_int = int('9'*4300)
    
    @lru_cache
    def safe_inv(f):
        if f == 0:
            return 1.0
        else:
            return 1.0 / f

    @lru_cache
    def gamma(i: list[float]) -> list[float]:
        """Apply gamma function to list of input variables."""
        def g(x: float) -> float:
            j = safe_inv(1 - (x**2 / C2))
            return np.sqrt(j)
        return map(g, i)
    
    # These are not capping velocity input but later cap velocity output.
    t_transform = [ti - ((vi*xi)/C2) for (ti,vi,xi) in zip(t,x,v)]
    x_transform = [xi - (vi*ti) for (ti,vi,xi) in zip(t,x,v)]
    
    dt = gamma(t_transform)
    dx = gamma(x_transform)
    dv = limit_delta(dx, dt)
    
    return (dx, dt, dv)
            



def new_path(x_list: list[float], t_list: list[float]):
    """
    Limit the magnitude of dx/dt to the speed of light.
    Then return a new x_list with the deltas added, and the same t_list
    """
    C = 299792458.0
    h = 6.62607015e-34 # Planck's constant value
    G = 6.6743e-11 # gravitational constant value
    planck_length = np.sqrt((h*G) / C**3)
    
    deltas = limit_delta(x_list, t_list)
    
    # keep values below C and above Planck length
    new_x_list = [cap_pl(cap_c(x+d)) for (x,d) in zip(x_list, deltas)] 
    
    # Apply Lorentz Boosts
    new_x_list, t_list, v_list = lorentz_boost(new_x_list, t_list, deltas)
    
    return new_x_list, t_list



def new_path_k(x_list: list[float], t_list: list[float], k_list: list[float]) -> list[float]:
    """
    Version that accepts momentum, zeroes out momentum when delta >= c, and returns new lists.
    """
    new_k_list = []
    C = 299792458.0
    x_grad = gradient(x_list)
    t_grad = gradient(t_list)
    d_list = x_grad / t_grad
    for (i,j) in zip(d_list, k_list):
        if i >= C:
            new_k_list.append(0.0)
        else:
            new_k_list.append(j)
    new_x_list, t_list = new_path(x_list, t_list)
    new_x_list, t_list, v_list = lorentz_boost(new_x_list, t_list, limit_delta(new_x_list, t_list))
    return new_x_list, new_k_list, t_list



def split_complex(values):
    """
    Splits a NumPy array into its real and imaginary parts.
    """
    import numpy as np
    real = [np.real(v) for v in values]
    imaginary = [np.imag(v) for v in values]
    return real, imaginary
    

def calculate_psi(time: float, xo: float, k: float, s: float) -> list[float]:
    """
    Adapted from https://www.taheramlaki.com/blog/articles/quantum-particles/
    xo <- initial wave packet's center, x0
    k <- initial momentum, k0
    s <- sigma
    Returns 1D NumPy ndarray of complex numbers.
    """
    x_min, x_max, num_points = -20, 20, 8000 # leads to an array of 8000 points evenly spaced in [-20, 20]
    grid = np.linspace(x_min, x_max, num_points) # this creates that array, it's a grid for e.g. matplotlib 
    s2 = s**2 # square of the sigma value
    # Computing the equation's return value has 3 steps
    psi = np.exp(-0.25 * np.square(grid - xo - 2j * k * s2) / (s2 + 1j * time)) # step 1
    psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * time) # step 2
    psi = np.power(0.5 * s2 / np.pi, 0.25) * psi # step 3
    return psi




def calculate_psi_marseille(time: list[float], xo: list[float], k: list[float], s: float) -> list[float]:
    """
    Apply some limits to the rate of change when given a list. Then calculate psi like normal.
    """
    x_min, x_max, num_points = -20, 20, 8000 # leads to an array of 8000 points evenly spaced in [-20, 20]
    grid = np.linspace(x_min, x_max, num_points) # this creates that array, it's a grid for e.g. matplotlib 
    s2 = s**2 # square of the sigma value
    # Computing the equation's return value has 3 steps
    def get_psi(grid, xo, k, s2, time):
        psi = np.exp(-0.25 * np.square(grid - xo - 2j * k * s2) / (s2 + 1j * time)) # step 1
        psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * time) # step 2
        psi = np.power(0.5 * s2 / np.pi, 0.25) * psi # step 3
        return psi
    deltas = limit_delta(xo, time)
    new_x_list, new_k_list, t_list = new_path_k(xo, k, time)
    new_x_list, t_list, v_list = lorentz_boost(new_x_list, t_list, deltas)
    psi_list = [get_psi(grid, x, k, s2, t) for (x,k,t) in zip(new_x_list, new_k_list, t_list)]
    return psi_list, psi_list[-1]



def psi_full(time, xo, k, s, x_min, x_max, num_points):
    """
    Adapted from animation demo by https://www.taheramlaki.com/blog/articles/quantum-particles/
    xo <- initial wave packet's center, x0
    k <- initial momentum, k0
    s <- sigma
    x_min, x_max, num_points <- grid spacing for output values, essentially a type of interpolation
    Returns NumPy ndarray of complex numbers.
    """
    grid = np.linspace(x_min, x_max, num_points) # this creates that array, it's a grid for e.g. matplotlib 
    s2 = s**2 # square of the sigma value
    # Computing the equation's return value has 3 steps, it's neater visually on a chalkboard
    psi = np.exp(-0.25 * np.square(grid - xo - 2j * k * s2) / (s2 + 1j * time)) # step 1
    psi = psi * np.exp(1j * k * xo - k**2 * s2) / np.sqrt(s2 + 1j * time) # step 2
    psi = np.power(0.5 * s2 / np.pi, 0.25) * psi # step 3
    return psi




def psi_variable_window(x_min, x_max, num_points):
    """What if we modulate the window size to see if it does machine learning?"""
    time, xo, k, s = 0, 0, 2, 1
    psi_full(time, xo, k, s, x_min, x_max, num_points)
    


def psi_partial(t, x):
    """
    Partial function over psi() when momentum and sigma are constant.
    Experiment using functools partial. 
    functools.partial eats a variable number of arguments from the end of the function.
    The variable number might be useful in high dimensionalities.
    """
    from functools import partial
    sigma = 1
    momentum = 2
    def flip_psi(s, k, x, t):
        """Literally just to inverse the order of inputs for the partial function."""
        return calculate_psi(t, x, k, s)
    part = partial(flip_psi, sigma, momentum) # still consumes x and t
    return part(x, t)

 
def psi_partial_k(t, x, k):
    """
    Partial function over psi() allowing variable momentum. 
    Only sigma is constant and hard-coded at 1.0
    """
    from functools import partial
    sigma = 1.0
    def flip_psi(s, k, x, t):
        return calculate_psi(t, x, k, s)
    part = partial(flip_psi, sigma) # sigma = 1, still consumes k (momentum), x (position), and time
    return part(k, x, t)


def psi_x_old(x):
    """
    Partial function over psi() allowing variable momentum. 
    Only sigma is constant and hard-coded at 1.0
    """
    from functools import partial
    sigma = 1.0
    t = np.arange(0, 10, len(x))
    k = 1.0
    def flip_psi(s, k, x, t):
        return calculate_psi(t, x, k, s)
    part = partial(flip_psi, sigma) # sigma = 1, still consumes k (momentum), x (position), and time
    return part(k, x, t)



def psi_x(x):
    """
    Partial function over psi() allowing variable momentum. 
    Only sigma is constant and hard-coded at 1.0
    """
    s = 1.0
    t = np.arange(0, 10, len(x))
    k = 1.0
    return calculate_psi(t, x, k, s)



def psi_matrix(t_list, x_list):
    """
    Consumes:
    * list of time values
    * list of x axis position values
    Does some extras:
    * Limit dx/dt value to c and recalculate path
    Returns:
    * NumPy matrix showing how the wave function's animation changes
    """
    new_x_list, t_list = new_path(x_list, t_list)
    matrix = np.arange(0)
    for (x,t) in zip(new_x_list, t_list):
        row = psi_partial(x,t)
        matrix = np.append(matrix, row)
    return matrix




def calculate_wave_packet_2d(time, x_c, kx, sx, y_c, ky, sy):
    psi_ = 0.25 * np.square(xg - x_c - 2j * kx * sx**2) / (sx**2 + 1j * time)
    psi_ = np.exp(1j*kx*xg - kx**2 * sx**2 - psi_) / np.sqrt(sx**2 + 1j * time)
    psi_x = np.power(sx**2/(2*np.pi), 0.25) * psi_
    psi_ = 0.25 * np.square(yg - y_c - 2j * ky * sy ** 2) / (sy ** 2 + 1j * time)
    psi_ = np.exp(1j * ky * yg - ky ** 2 * sy ** 2 - psi_) / np.sqrt(sy ** 2 + 1j * time)
    psi_y = np.power(sy ** 2 / (2 * np.pi), 0.25) * psi_
    return psi_x * psi_y


def calculate_psi_2d(time, x_c, kx, sx, y_c, ky, sy):
    # return calculate_wave_packet(t, xo, kxo, sigma_x, yo, kyo, sigma_y) \
    #       + calculate_wave_packet(t, -xo, -kxo, sigma_x, yo, kyo, sigma_y)

    psi_ = np.zeros((num_points_x, num_points_y), dtype=np.complex128)
    for i in range(6):
        phi = 2 * np.pi * (i - 1) / 6
        xc = x_c * np.cos(phi) - y_c * np.sin(phi)
        yc = -x_c * np.sin(phi) + y_c * np.cos(phi)
        kx_ = kx * np.cos(phi) - ky * np.sin(phi)
        ky_ = -kxo * np.sin(phi) + ky * np.cos(phi)
        psi_ += calculate_wave_packet(time, xc, kx_, sx, yc, ky_, sy)
    return psi_