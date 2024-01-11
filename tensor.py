import random
import math

def normalize_ms(n)->float:
    """
    Meters per second version of normalizing.
    
    In: m/s
    Out: c unit vector
    Normalize number input in to speed of light unit vector.
    
    Args:
        n: number
        
    Returns:
        m: float
    """
    C = 299792458
    if n > C:
        m = 1.0
    elif n < 0:
        m = 0.0
    else:
        m= n / C
    return m


def normalize(v: tuple) -> tuple:
    vector = list(v)
    vector = [float(f)/max(vector) for f in vector]
    return tuple(vector)


def check_four(v: tuple) -> tuple:
    """
    The 4-vector (x,y,z,t) must sum to 1.0 per relativity.
    
    Args:
        v (tuple[float]): 4-vector of (x,y,z,t) which must sum to 1

    Returns:
        tuple: 4-vector renormalized to sum to 1
    """
    x, y, z, t = v[0], v[1], v[2], v[3]
    
    if (sum(v) > 1):
        if max(v) == t:
            x, y, z, t = 0.0, 0.0, 0.0, 1.0
            outvec = (x,y,z,t)
        else:
            outvec = normalize(v)
    else:
        outvec = normalize(v)
        
    return outvec