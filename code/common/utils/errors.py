class Dataset_Exception(Exception):
    def __init__(self, message):
        print("Excepcion en la generacion de Dataset")
        super().__init__(self, message)
        
        
class FileNotFound(Dataset_Exception):
    def __init__(self, message):
        super().__init__(self, message)
