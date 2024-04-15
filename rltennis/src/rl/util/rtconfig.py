import logging
import logging.config
import json
import os

def tc_runtime_configure(cfg_spec):
    """ 
    Configure an application's runtime configuration.
    
    Logging Guidelines: Store a logger in a class variable _logger and
    use 
    
    .. code-block:: python
    if (JsonCodecTest._logger.isEnabledFor(logging.INFO)):
    JsonCodecTest._logger.info("On Serialization: {}".format(serObj))

    """
    log_file = cfg_spec.get("logConfig", None)
    if (os.path.isfile(log_file)):
        with open(log_file, 'rt') as log_file_handle:
            config = json.load(log_file_handle)
        logging.config.dictConfig(config)
        #print("Loaded configuration from {}".format(log_file))
    else:
        log_level = cfg_spec.get("log_level", "WARNING")
        log_level = logging.getLevelName(log_level) 
        logging.basicConfig(level=log_level)
        
def tc_closest_filepath(source_path, target_file="logConfig.json", ancestor_levels=100):
    """
    Find file closest to the specified source path.
    
    :param str source_path: Path relative to which the closest target file's path is sought. Path may be of a file or directory.
    :param str target_file: Name of target file whose closest path is sought. Target may be a file or directory
    :param int ancestor_levels: Maximum levels to climb up the ancestors.
    :return Absolute path of the closest filepath for the target file to the source path.
    :rtype str
    """
    source_path = os.path.abspath(source_path)
    if os.path.isfile(source_path):
        curr_dir = os.path.dirname(source_path)
    elif not os.path.exists(source_path):
        return None
    else:
        curr_dir = source_path
    
    depth = ancestor_levels
    while (depth > 0):
        cfg_path = os.path.sep.join([curr_dir, target_file])
        if os.path.exists(cfg_path):
            return(cfg_path)
        curr_dir = os.path.dirname(curr_dir)
        depth = depth - 1
    return(None)