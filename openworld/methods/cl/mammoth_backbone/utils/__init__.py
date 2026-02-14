import inspect
import sys
import logging
from typing import Callable, Type, TypeVar, Union, get_args

T = TypeVar("T")


def infer_args_from_signature(signature: inspect.Signature, excluded_signature: inspect.Signature = None) -> dict:
    """
    Load the arguments of a function from its signature.

    Args:
        signature: the signature of the function

    Returns:
        the inferred arguments
    """
    excluded_args = {} if excluded_signature is None else list(excluded_signature.parameters.keys())
    parsable_args = {}

    for arg_name, value in list(signature.parameters.items()):
        if arg_name in excluded_args:
            continue
        if arg_name != 'self' and not arg_name.startswith('_'):
            default = value.default
            tp = str
            if value.annotation is not inspect._empty:
                tp = value.annotation
            elif default is not inspect.Parameter.empty:
                tp = type(default)
            if default is inspect.Parameter.empty and arg_name != 'num_classes':
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': True
                }
            else:
                parsable_args[arg_name] = {
                    'type': tp,
                    'required': False,
                    'default': default if default is not inspect.Parameter.empty else None
                }
    return parsable_args


def register_dynamic_module_fn(name: str, register: dict, tp: Type[T]):
    """
    Register a dynamic module in the specified dictionary.

    Args:
        name: the name of the module
        register: the dictionary where the module will be registered
        cls: the class to be registered
        tp: the type of the class, used to dynamically infer the arguments
    """
    name = name.replace('_', '-').lower()

    def register_network_fn(target: Union[T, Callable]) -> T:
        # check if the name is already registered
        if name in register:
            raise ValueError(f"Name {name} already registered!")

        # check if `cls` is a subclass of `T`
        if inspect.isfunction(target):
            signature = inspect.signature(target)
        elif isinstance(target, tp) or issubclass(target, tp):
            signature = inspect.signature(target.__init__)
        else:
            raise ValueError(f"The registered class must be a subclass of {tp.__class__.__name__} or a function returning {tp.__class__.__name__}")

        parsable_args = infer_args_from_signature(signature)
        register[name] = {'class': target, 'parsable_args': parsable_args}
        return target

    return register_network_fn
