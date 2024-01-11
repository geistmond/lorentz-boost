#!/usr/bin/python3
from typing import Any, Callable


class Functor:
    def __init__(self, value: Any) -> None:
        self.value = value
        
    def map(self, func: Callable[[Any], Any]) -> "Functor":
        return Functor(func(self.value))


class StringFunctor:
    """
    I've been working with base64 encoded images. Those are string handling.
    I feel like having some higher-order string handling operators for personal use anyway.
    """
    from typing import Callable
    def __init__(self, value: str):
        self.value = value
    
    def map(self, func: Callable[[str], str]):
        return StringFunctor([func(self.value)])
    

class ListFunctor:
    def __init__(self, values: list[Any]):
        self.values = values
    
    def map(self, func: Callable[[Any], Any]):
        return ListFunctor([func(value) for value in self.values])
    
    
class StringToListFunctor:
    def __init__(self, value: str):
        self.value = value
    
    def map(self, func: Callable[[str], str]):
        return StringFunctor([func(self.value)])
    