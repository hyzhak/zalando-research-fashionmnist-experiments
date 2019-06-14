def params_to_filename(d):
    return '_'.join(f'{key}={value}' for key, value in d.items()) or 'default'
