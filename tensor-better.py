from dataclasses import dataclass 

import math

# prettier debugging
from icecream import ic

"""
Test project to build back some of my understanding of physics through a different persona.

*** WHAT I AM DOING IN THIS IS RE-IMPLEMENTING GEOMETRY AS CLASSES FOR RELATIVITY. ***

Considerations:

* Starting point is Inertial Frame of Reference

* No grid system or coordinates, just velocity components that sum to 1.0 (normalized to C)

* Distance is relative! This means there needs to be a distance metric that inherits frames of reference. It is just a radius.

* What does it mean for a direction to have two sides? Left/Right dichotomy not there in a time-type axis even with time travel?

* What does it mean for an axis to be Left and Up rather than Left and Right or Up and Down? 

* Normalizing everything to account for speed of light limitations

* 90 degree angle is where coordinate systems emerge by way of perpendicularity, is 45 half on each axis?


Could approach like this:

Arrow class is a velocity Float in the unit vector [0.0, 1.0] and a string identifier

Angle class is a Float in the unit vector [0.0, 1.0] with radian and degree normalizations

Direction class is: Arrow 1, Arrow , Angle between the arrows (usually 0.25 for right angle)

Space class is 3 directions for 3D

"""



def normalize(vector: list) -> list:
    vector = [float(f)/max(vector) for f in vector]
    return vector
    

@dataclass
class Arrow:
    """
    Defines a velocity measurement.
    0.0 = no velocity here
    1.0 = speed of light
    """
    v: float
    
    def velocity(self) -> float:
        if self.v > 1.0:
            self.v = 1.0
            return self.v
        elif self.v < 0.0:
            self.v = 0.0
            return self.v
        else:
            return self.v
        
        
@dataclass
class Angle:
    """
    Angle is normalized to the unit vector. It wraps.
    """
    a: float

    def phi(self) -> float:
        self.a = self.a % 1.0
        ic(self.name, " Phi: ", self.a)
        return self.a
    
    def radians(self) -> float:
        f = self.phi()
        radians = f * 2 * math.pi
        ic(self.name, " Radians: ", radians)
        return radians
    
    def degrees(self) -> float:
        f = self.phi()
        degrees = f * 360.0
        ic(self.name, " Degrees: ", degrees)
        return degrees
         
        
    
@dataclass
class Direction:
    """
    Direction consists of:
    * Name
    * Arrow 1
    * Arrow 2
    * Angle between them
    """
    a1: Arrow(0.0)
    a2: Arrow(0.0)
    ang: Angle
    
    def move(self, acc=1.0):
        p = self.ang.phi()
        self.a1.v = self.a1.velocity()
        if self.a1.v > 0:
            v1 = (self.a1.v * acc) % 1.0
            v2 = 1 - v1
            self.a1.v = v1
            self.a2.v = v2
        elif self.a2.v > 0:
        
        
    

@dataclass
class Line:
    """
    A line in Euclidian space is 2 directions at 180 degrees.
    Motion is limited to one or the other.
    Positive numbers -> add to d1
    Negative numbers -> add to d2
    """
    d1: Direction
    d2: Direction
    
    def value(self) -> float:
        # First direction, retrieve and normalize velocity, set angle to 180deg
        self.d1.a1.v = self.d1.a1.velocity()
        self.d1.a2.v = self.d1.a2.velocity()
        self.d1.ang = Angle(0.5)
        self.d1.ang.phi() 
        
        # Second direction, same process
        self.d2.a1.v = self.d2.a1.velocity()
        self.d2.a2.v = self.d2.a2.velocity()
        self.d2.ang = Angle(0.5)
        self.d2.ang.phi()
    
    def move(self, rate):
        if rate > 0.0:
            self.d1.a1.velocity()
        elif rate < 0.0:
        else:
        

@dataclass
class SignedUnitVector:
    f: float
    def value(self) -> float:
        if self.f > 1.0:
            self.f = 1.0
            return self.f
        elif self.f < -1.0:
            self.f = -1.0
            return self.f
        else:
            return self.f


@dataclass
class UnsignedUnitVector:
    f: float
    def value(self) -> float:
        self.f = abs(self.f)
        if self.f > 1.0:
            self.f = 1.0
            return self.f
        else:
            return self.f
    
    
@dataclass
class UnsignedUnitVectorSpace:
    space: list[UnsignedUnitVector]
    def value(self) -> list:
        values = []
        for s in self.space:
            v = s.value()
            values.append(v)
        normed = normalize(values)
        return normed
    
@dataclass
class SignedUnitVectorSpace:
    space: list[SignedUnitVector]
    def value(self) -> list:
        values = []
        for s in self.space:
            v = s.value()
            values.append(v)
        normed = normalize(values)
        return normed