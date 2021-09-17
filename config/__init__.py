import collections
import os
import yaml


class ConfigError(Exception):
    pass


def load_config(env, alg):
    """Load a configuration file.
    """

    res = None

    for path in __config_file_paths(env, alg):
        new_config = None
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as config:
                    new_config = yaml.safe_load(config.read())
            except (yaml.parser.ParserError, yaml.parser.ScannerError) as err:
                raise ConfigError('Config file %s: failed to parse: %s' % (path, err))
        if res is None:
            if new_config is None:
                raise ConfigError('Base configuration file %s not found.' % (path))
            res = new_config
        elif new_config is not None:
            __update_dict(res, new_config)

    return res


def __config_file_paths(env, alg):
    """
    Paths in which to look for config files, by increasing order of
    priority (i.e., any config in the last path should take precedence
    over the others).
    """

    paths = [
        os.path.join(os.path.dirname(__file__), 'default.yaml'),
    ]
    if env is not None:
        paths.append(os.path.join(os.path.dirname(__file__), 'envs', env + '.yaml'))
    if alg is not None:
        paths.append(os.path.join(os.path.dirname(__file__), 'algs', alg + '.yaml'))
    return paths



def __update_dict(orig, update):
    """Deep update of a dictionary
    For each entry (k, v) in update such that both orig[k] and v are
    dictionaries, orig[k] is recurisvely updated to v.
    For all other entries (k, v), orig[k] is set to v.
    """
    for (key, value) in update.items():
        if (key in orig and
            isinstance(value, collections.Mapping) and
                isinstance(orig[key], collections.Mapping)):
            __update_dict(orig[key], value)
        else:
            orig[key] = value


class Config(object):

    def __init__(self, env, alg):
        for key, val in load_config(env, alg).items():
            self.__dict__[key] = val
