class Register:
    def __init__(self, name: str):
        self.name = name
        self.entries = {}

    def register(self, key: str):
        def decorator(class_builder):
            self.entries[key] = class_builder
            return class_builder

        return decorator

    def build(self, type: str):
        if type not in self.entries:
            raise ValueError(
                f"Entry {type} not found in registry {self.name}. Available entries: {', '.join(self.entries.keys())}"
            )
        return self.entries[type]
