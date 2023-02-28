import numpy as np

class State(object):
    def __init__(self, properties: dict = {}, default: dict = {}):
        N = sum(properties.values())
        self._data = np.zeros(N)
        self._meta = {}

        start_index = 0
        for prop, size in properties.items():
            default_value = default[prop] if prop in default else 0.0

            setattr(self, prop, self._data[start_index:start_index + size])
            getattr(self, prop)[:] = default_value
            
            self._meta[prop] = {
                'index':    start_index,
                'size':     size,
                'default':  default_value
            }

            start_index += size

        # impl copy function, using current state as default, when reset
        self.copy = lambda: State(properties, dict([(prop, self._data[meta['index']:meta['index']+meta['size']]) for prop, meta in self._meta.items()]))

    def __repr__(self):
        return '\n'.join([f"{prop}[{meta['size']}]: {tuple(getattr(self, prop)[:])}" for prop, meta in self._meta.items()])
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._data[index]
        elif isinstance(index, str):
            return getattr(self, index)[:]
        else:
            raise Exception("Invalid subscript.")

    def __setitem__(self, index, value):
        if isinstance(index, int):
            self._data[index] = value
        elif isinstance(index, str):
            getattr(self, index)[:] = value
        else:
            raise Exception("Invalid subscript.")

    def has(self, prop: str) -> bool:
        try:
            getattr(self, prop)
            return True
        except:
            return False
            
    def reset(self):
        for prop, meta in self._meta.items():
            getattr(self, prop)[:] = meta['default']

    def update(self, properties: dict = {}):
        for prop, value in properties.items():
            getattr(self, prop)[:] = value