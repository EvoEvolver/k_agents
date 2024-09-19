

class Singleton(object):
    """
    The singleton class is used to make sure that there is only one instance of the class.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of the class if it doesn't exist. Otherwise return the existing instance.

        Parameters:
            args (list): The arguments to pass to the class constructor.
            kwargs (dict): The keyword arguments to pass to the class constructor.

        Returns:
            object: The instance of the class.
        """
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the singleton class.
        """
        if self._initialized:
            return

        self._initialized = True