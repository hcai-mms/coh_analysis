def str_to_bool(string):
    string = string.lower()
    if string == 'true':
        return True
    if string == 'false':
        return False
    return None