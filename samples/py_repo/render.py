"""Rendering of results."""


class Renderer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = []

    def draw(self, items):
        self.clear()
        for item in items:
            self.buffer.append(self.fmt(item))
        return "\n".join(self.buffer)

    def clear(self):
        del self.buffer[:]

    def fmt(self, item):
        return str(item)
