class MalformedExperimentError(Exception):
    """Raises if the generator or the metamodel do not fit together. This may happen if you use dummies or perfects incorrectly."""