# TODO: Need to figure out how this process works
'''Implements grid-search over training configurations'''
from collections import namedtuple
from copy import deepcopy
import os
import os.path
import yaml

Parameter = namedtuple("Parameter", ["key", "value"])

def get_parameters(dictionary, base_key=[]):
    '''Recursively searches through a dictionary and returns all tunable parameters'''

    if "grid_search" in dictionary:
        assert len(dictionary) == 1, "'grid_search' entries must be unique"
        assert isinstance(dictionary["grid_search"], list), "'grid_search' value must be a list of parameters"

        return [Parameter(base_key, dictionary["grid_search"])]
    else:
        parameters = []

        for key, value in dictionary.items():
            if isinstance(value, dict):
                parameters += get_parameters(value, base_key + [key])
        
        return parameters


def set_recursive(dictionary, key, value):
    '''Sets a value in a nested dictionary'''

    for idx in range(len(key) - 1):
        dictionary = dictionary[key[idx]]

    dictionary[key[-1]] = value


def serialize_dict(dictionary):
    '''converts a nested dict to a string'''

    items = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            items.append(f"{key}={serialize_dict(value)}")
        else:
            items.append(f"{key}={value}")
    
    return f"({'.'.join(items)})"


def get_variations(base_name, base_config, free_parameters, set_parameters=[]):
    '''Takes a list of tunable parameters and generates a list of configurations'''

    if len(free_parameters) == 0:
        name = []

        for param in set_parameters:
            value = param.value
            if isinstance(value, dict):
                value = serialize_dict(value)
            name.append(f"{param.key[-1]}={value}")

        name = ','.join(name)
        name = f"{base_name}_{name}"

        config = deepcopy(base_config)

        for p in set_parameters:
            set_recursive(config, p.key, p.value)

        return {name: config}
    else:
        variations = {}

        for value in free_parameters[0].value:
            parameter = Parameter(free_parameters[0].key, value)
            variations.update(get_variations(base_name, 
                                             base_config, 
                                             free_parameters=free_parameters[1:],
                                             set_parameters=set_parameters + [parameter]))

        return variations

# Copied from 'run.py' to avoid circular dependency
def make_or_use_dir(path, tag):
    sub_path = os.path.join(path, tag)
    if not os.path.exists(sub_path):
        os.makedirs(sub_path)
    return sub_path

def save_variations(base_name, base_config, free_parameters, set_parameters=[]):
    '''Takes a list of tunable parameters and generates a list of configurations'''

    path = make_or_use_dir("./experiments", base_name)

    if len(free_parameters) == 0:
        name = []

        for param in set_parameters:
            value = param.value
            if isinstance(value, dict):
                value = serialize_dict(value)
            name.append(f"{param.key[-1]}={value}")

        name = ','.join(name)
        name = f"{base_name}_{name}"

        config = deepcopy(base_config)

        for p in set_parameters:
            set_recursive(config, p.key, p.value)

        return {name: config}
    else:
        variations = {}

        for value in free_parameters[0].value:
            parameter = Parameter(free_parameters[0].key, value)
            variations.update(save_variations(base_name, 
                                             base_config, 
                                             free_parameters=free_parameters[1:],
                                             set_parameters=set_parameters + [parameter]))

        idx = 0
        for variation_name, config in variations.items():   
            variation_name = variation_name.replace("=", "").replace("," , "_")
            variation_config_file = os.path.join(path,"config_{}_experiment{}.yaml".format(base_name, idx))
            idx += 1
            
            with open(variation_config_file, 'w') as config_file:
                yaml.dump({variation_name: config}, config_file)

        return variations
                                     
def grid_search(name, config):
    parameters = get_parameters(config)

    if len(parameters) == 0:
        return None
    else:
        parameters.sort(key=lambda param: param.key[-1])
        return get_variations(name, config, parameters)

def generate_config_files(name, config):
    parameters = get_parameters(config)
    assert len(parameters) != 0, "There must be tunable parameters for generating config files!"
    parameters.sort(key=lambda param: param.key[-1])
    return save_variations(name, config, parameters)
